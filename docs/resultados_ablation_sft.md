# Resultados de la ablation de SFT — evaluación completa

**Fecha:** 2026-07-07 · **Autor:** reconstruido y ejecutado con Claude Code
**Alcance:** evaluación generativa de las 4 variantes de loss + baseline base sobre el
test completo (1263 recetas) del dataset `final-10k`.

Este documento es el registro **completo y autocontenido** del experimento: qué se
corrió, cómo, con qué números, qué significan y qué tan bueno es el modelo. La bitácora
operativa (`docs/codigo/bitacora_experimentos.md`) tiene el historial cronológico; este
archivo es el informe de resultados.

---

## 1. Resumen ejecutivo

- **El SFT es necesario y funciona.** El modelo base (Qwen3-4B-Thinking, sin adapter)
  saca **0.0%** de cobertura en todas las métricas; no produce conceptos sino
  fragmentos de razonamiento. El SFT lo lleva a producir conceptos on-task.
- **Ganó `soft_ce` (dataset × expected_logprob).** Lidera 6 de 7 métricas de cobertura
  y es el menos verboso.
- **El eje decisivo es la *agregación de candidatos*, no el *weighting*.**
  `expected_logprob` duplica a `logsumexp` (~7.5% vs ~4% top-1 exacto). Ponderar por
  dataset vs uniforme casi no mueve la aguja.
- **El modelo entiende la tarea, pero le falta forma.** La cobertura exacta (7%) es baja,
  pero la semántica (34% top-1 / 51% any@k) es ~5× mayor: el conocimiento composicional
  está; el cuello de botella es la **verbosidad** y algunos **tokens corruptos**.
- **El CCS de embeddings contradijo a cobertura** y por eso queda como
  secundario/exploratorio, no decisorio (es un proxy geométrico, no un juicio de
  creatividad; ver §11.4).
- **Nueva métrica de creatividad con juez LLM (§12), corrida completa.** Un juez fuerte
  (Gemini 2.5 Pro, batch), sin ver las respuestas del teacher, separa **Validez**
  (correcto) de **Creatividad** (correcto **y** nuevo). Sobre las 1263 recetas de
  `soft_ce`: **Validez 83.3% top-1 / 96.8% any@k**, **Creatividad 53.6% / 76.9%**. Es
  ~11× la cobertura exacta (7.3%): el modelo es mucho mejor de lo que el exact-match
  sugería, y **más de la mitad de sus top-1 son descubrimientos** correctos fuera de la
  lista del teacher. (Sonnet quedó descartado: la cuenta de créditos no puede comprar
  Claude en Marketplace, §12.4.)

---

## 2. Contexto y objetivo

El proyecto destila *creatividad composicional* estilo *Infinite Craft* en un LLM chico:
dado un par de conceptos (`input_a`, `input_b`), producir el concepto resultante
(`fire + water → steam`). Se entrenó Qwen3-4B-Thinking con QLoRA bajo una **familia de
loss unificada** parametrizada por dos ejes:

- `candidate_weighting`: `uniform` | `dataset` (pesos por candidato desde los datos).
- `candidate_aggregation`: `expected_logprob` (soft-CE) | `logsumexp_prob` (concept-set).

Las 4 combinaciones se entrenaron (misma data, mismo prompt, misma config salvo los dos
ejes). **La pregunta central del experimento:** ¿cuál de las 4 losses produce el mejor
modelo generativo? Hasta ahora solo se había evaluado 1 de las 4; este experimento cierra
la ablation y agrega el baseline del modelo base.

---

## 3. Metodología

### 3.1 Modelos evaluados

| Etiqueta | weighting × aggregation | `run_dir` (en `gs://llm-craft-bucket/runs/`) |
|---|---|---|
| `concept_set` | dataset × logsumexp | `2026-07-01_1950_qwen3_4b_thinking_10k` |
| `soft_ce` | dataset × expected_logprob | `2026-07-02_2300_qwen3_4b_thinking_10k_softce` |
| `concept_set_uniform` | uniform × logsumexp | `2026-07-03_2250_qwen3_4b_thinking_10k_concept_set_uniform` |
| `ce_uniform` | uniform × expected_logprob | `2026-07-03_2255_qwen3_4b_thinking_10k_ce_uniform` |
| `base` | — (sin adapter) | modelo `Qwen/Qwen3-4B-Thinking-2507` crudo |

Base model: `Qwen/Qwen3-4B-Thinking-2507`, cargado en 4-bit (QLoRA). Los 4 adapters
comparten datos (train SHA `3399f18…`, 10.165 filas; dev `cd6c911…`) y difieren **solo**
en el eje de loss (ablation limpia, verificada en el diff de `losses.py`).

### 3.2 Datos de evaluación

`datasets/final-10k/test.jsonl`, **1263 recetas**. Splits asignados por hash del par de
inputs ⇒ ningún par de test aparece en train/dev. Cada receta trae `known_outputs` (lista
de salidas observadas/plausibles del teacher) y `canonical_output` (la principal).

### 3.3 Configuración de generación (idéntica para las 5 evals)

`num_samples=4`, `max_new_tokens=8`, `temperature=0.6`, `top_p=0.9`, `top_k=40`,
`repetition_penalty=1.15`, `no_repeat_ngram_size=3`, `max_concept_words=3`. Prompt del
student = el simple de `collator.py` (system "You combine two concepts…" + user "Given
two concepts, combine them…"). Igual en train y eval (regla dura).

**Manejo del thinking.** Qwen3-Thinking inyecta `<think>` siempre. Los 4 SFT aprendieron a
responder directo (se supervisaron solo sobre el concepto). El **base** se evaluó con
`--close_think_prompt` (precerrar `<think></think>` para forzar respuesta directa) +
`--strip_think` — el trato más favorable posible al base.

### 3.4 Métricas

- **Cobertura exacta (primaria).** `top1/any@k` contra `known_outputs` y contra
  `canonical_output`, con normalización (minúsculas, espacios). `metrics.py`.
- **Cobertura semántica (secundaria, robustez).** Coseno ≥0.75 entre la muestra y algún
  `known_output`, embeddings `all-MiniLM-L6-v2`.
- **Verbosidad.** Media de palabras del top-1 y fracción ≤2 palabras (la tarea es ≤2).
- **CCS (secundaria/exploratoria).** `C(x)=α·mean(q^λ·n)+(1−α)·d`, α=0.8, λ=2.

El scoring de cobertura es **offline en CPU** (`score_coverage.py`), separado de la
generación en GPU: una vez guardado `predictions.jsonl`, cualquier métrica se recalcula
gratis.

---

## 4. Infraestructura: la crisis de GPU y su fix

### 4.1 El problema

Al correr las evals, los nodos **T4 y A100 caían a CPU** (una eval de 4B tarda horas);
solo **L4** usaba GPU, pero tenía poca capacidad (jobs encolados).

### 4.2 Causa raíz

El wheel default de `torch 2.12.1` en PyPI trae runtime **CUDA 13** (`nvidia-*-cu13`), que
exige driver CUDA 13. Los nodos T4/A100 de Vertex en `us-central1` tienen driver **CUDA
12.2** (`found version 12020`) — un *major* por detrás ⇒ sin compatibilidad ⇒ torch no ve
la GPU y cae a CPU. Los L4 tenían driver más nuevo, por eso eran los únicos que andaban.

### 4.3 El fix

Se ancló torch al índice **cu126** de PyTorch, solo en Linux (macOS sigue con el wheel de
PyPI para no romper dev local ni `uv lock`):

```toml
# pyproject.toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux'" }]
```

cu126 corre sobre la familia de driver R525+, que el driver 12.2 (R535) cumple (CUDA
*minor-version compatibility*). Se reconstruyó la imagen; un smoke en A100 80GB confirmó
`Model ready on device cuda:0`. Esto **desbloqueó A100 y T4** además de L4.

### 4.4 Otros arreglos de la ronda

- **Cuota A100 80GB `us-central1`: 1 → 4** (Cloud Quotas API), para correr en paralelo.
- **`vertex_submit --no-wait`**: `job.submit()` async en vez de `job.run()` bloqueante,
  para que un proceso local caído no sea single-point-of-failure de un batch.
- **Staging sin checkpoints**: pasar el `run_dir` como ruta del mount `/gcs/…` lee el
  adapter directo (sin copiar los ~115 GiB de checkpoints), eliminando el fallo de disco.
- **Lección de quoting** ⚠️: al reenviar flags extra por `-- …`, pasarlos como **tokens
  separados** (o array bash `"${GEN[@]}"`), no como string sin comillas. Dos jobs
  fallaron porque los 9 flags de decoding llegaron como **un solo token** argv.

---

## 5. Resultados cuantitativos

### 5.1 Cobertura (métrica primaria)

| Modelo | top1_known | any@k_known | top1_canon | any@k_canon |
|---|---|---|---|---|
| **`soft_ce`** | 0.073 | **0.143** | **0.042** | **0.073** |
| `ce_uniform` | **0.075** | 0.134 | 0.038 | 0.064 |
| `concept_set` | 0.040 | 0.077 | 0.026 | 0.041 |
| `concept_set_uniform` | 0.038 | 0.075 | 0.025 | 0.041 |
| **base** | 0.000 | 0.000 | 0.000 | 0.000 |

`known` = coincide con cualquier salida conocida; `canon` = coincide con la principal.
`any@k` = alguna de las 4 muestras acierta.

### 5.2 Cobertura semántica (coseno ≥0.75, MiniLM)

| Modelo | top1_sem | any@k_sem |
|---|---|---|
| **`soft_ce`** | **0.337** | **0.511** |
| `ce_uniform` | 0.290 | 0.476 |
| `concept_set` | 0.274 | 0.410 |
| `concept_set_uniform` | 0.259 | 0.394 |
| base | 0.000 | 0.000 |

### 5.3 Verbosidad

| Modelo | media palabras (top-1) | ≤2 palabras |
|---|---|---|
| `concept_set_uniform` | 2.78 | 18.2% |
| `concept_set` | 2.71 | 24.0% |
| `ce_uniform` | 2.59 | 33.7% |
| **`soft_ce`** | **2.50** | **40.9%** |
| base | 2.99 | 1.3% |

### 5.4 CCS (secundaria / exploratoria)

> ⚠️ **El CCS de esta tabla es un proxy geométrico de embeddings, NO un juicio de
> creatividad.** Sus tres componentes (`plaus`, `novelty`, `diversity`) son distancias
> coseno con `all-MiniLM-L6-v2`, sin ningún modelo que juzgue si la respuesta es
> correcta. Ver el glosario (§11) para qué mide exactamente cada componente y sus
> límites, y §12 para la métrica de creatividad basada en juez LLM que la reemplaza.

| Modelo | CCS | plaus | novelty | diversity |
|---|---|---|---|---|
| `concept_set` | 0.324 | 0.780 | 0.304 | 0.887 |
| `concept_set_uniform` | 0.321 | 0.776 | 0.306 | 0.877 |
| `soft_ce` | 0.318 | 0.786 | 0.303 | 0.850 |
| `ce_uniform` | 0.313 | 0.781 | 0.301 | 0.841 |
| base | 0.284 | 0.600 | 0.409 | 0.829 |

### 5.5 Lecturas

1. **`soft_ce` gana** en cobertura exacta (salvo top1_known, empate técnico con
   `ce_uniform` a 0.2 pp), semántica (todas) y brevedad. Es el ganador claro.
2. **La agregación manda.** `expected_logprob` (~7.5% top1) ≈ 2× `logsumexp` (~4%).
   Intuición: soft-CE promedia la NLL por candidato ⇒ enseña la respuesta *modal*; el
   log-sum-exp reparte masa sobre el conjunto y **diluye el top-1**. El weighting
   (dataset vs uniform) es un efecto de segundo orden.
3. **El base es piso 0.** Cero coincidencias exactas y semánticas; ni siquiera con el
   mejor trato del thinking. El SFT no "mejora" al base: lo **habilita**.
4. **CCS contradice a cobertura.** Rankea `concept_set` primero (premia diversidad sin
   verificar correctitud) — al revés que cobertura. Además la banda entre las 4 celdas
   (0.313–0.324) es ruido. ⇒ CCS no sirve todavía para elegir loss; queda como línea de
   investigación.

---

## 6. Análisis cualitativo de las recetas (modelo ganador `soft_ce`)

Estadísticas sobre las 1263 recetas (top-1 de cada una):

- **Distribución de largo:** 1 palabra 115 (9%), 2 palabras 402 (32%), 3 palabras 746 (59%).
- **Colapso de modo** (las 4 muestras idénticas): 172 (13.6%).
- **Eco de inputs** (el top-1 contiene una palabra de input): 545 (43.2%).
- **Diversidad global:** 1218 top-1 distintos (96.4%) — no repite la misma respuesta entre recetas.

### 6.1 Aciertos y cuasi-aciertos (el modelo entiende)

Muchos "fallos" de exact-match son en realidad **correctos**:

| Receta | known | predicción | juicio |
|---|---|---|---|
| big + ice | iceberg, glacier | **iceberg** | exacto |
| paper + phone | faxmachine | **fax machine** | exacto (solo el espacio) |
| beehive + farmer | beekeeper, apiary | apiary owner / beekeeper farm manager | correcto |
| element + skull | calcium, bone | calcium phosphate | correcto (¡mejor que el label!) |
| electricity + storm | lightning | lightning rod system | correcto, sobre-especificado |
| cell + glasses | microscope | microscope lens correction | correcto |
| camera + paper | film paper | photo paper | correcto |
| land + organic matter | soil, compost | soil fertility | correcto |
| earth + fire | lava, magma | lava rock | correcto |
| china + fire | chinese lantern | chinese lantern festival | correcto |

Esto explica la brecha 7% exacto → 34-51% semántico: **el conocimiento composicional está**.

### 6.2 Problemas de forma (no de comprensión)

1. **Verbosidad / sobre-especificación** (el problema #1). Dice "strawberry jam jar" en
   vez de "strawberry", "lightning rod system" en vez de "lightning". El 59% son de 3
   palabras. Penaliza directo el exact-match, que quiere el concepto **corto y canónico**.
2. **Tokens corruptos / derailments.** Salidas mal formadas: `snow globe orbiterator`,
   `mine cartwheelbarrow wheelbarrow`, `scalpel handlebar mustache`, `vault wallroom
   door`, `tethered bone a`, `owl garden a`. Parecen artefactos del presupuesto de 8
   tokens cortando a mitad de palabra y/o del `repetition_penalty`/`no_repeat_ngram`.
3. **Colapso de modo (13.6%)** — 4 muestras idénticas (`moldy bread` ×4, `calcium
   phosphate` ×4). Reduce el beneficio de `any@k`.
4. **Misses genuinos** (minoría): `food+time → "moldy bread"` (esperaba fastfood/mealtime),
   `cell+devolution → "organelle evolution"` (dirección equivocada).

### 6.3 Base vs SFT (por qué el base saca 0)

Con el mejor trato del thinking, el base produce **meta-comentario**, no conceptos:

| Receta | base | soft_ce |
|---|---|---|
| cream + ice | "Hmm the user" | frozen cream / ice cream pop |
| castle wall + castle wall | "Hmm the user" ×4 | castle fortress complex |
| energy + tree | "We are combining" | **photosynthesis** (exacto) |
| hut + steam | "Okay so I" | sauna hut / sauna room |

---

## 7. Evaluación: ¿es bueno el modelo?

**Como motor de asociación composicional: sí, es sorprendentemente bueno.** Entiende cómo
se combinan dos conceptos y produce resultados semánticamente aptos, a veces tan buenos o
mejores que la referencia (`element+skull → calcium phosphate`). La cobertura semántica
(34% top-1, 51% any@k sobre 1263 pares no vistos) y la diversidad global (96% de top-1
distintos) lo confirman.

**Como generador del formato que la tarea premia: todavía no.** La cobertura exacta (7%)
es baja porque el exact-match castiga la verbosidad y la elección de sinónimo. El modelo
sabe *qué* responder pero no *cómo* de corto y canónico. Sumado a los tokens corruptos y
al colapso de modo, la "calidad de producto" no está.

**Veredicto:** el SFT logró lo difícil (enseñar la asociación) y dejó lo fácil-pero-clave
(brevedad y formato) sin resolver. La brecha exacto↔semántico dice exactamente dónde
invertir: **no en más capacidad de razonamiento, sino en control de forma** (brevedad,
decoding, y preferencia por el concepto canónico).

---

## 8. Limitaciones y amenazas a la validez

- **Labels ruidosos.** Algunos `known_outputs` del teacher son raros (`bones+rope →
  "needleandthread"`), así que el exact-match subestima el desempeño real.
- **`max_new_tokens=8`** probablemente **causa** parte de los tokens corruptos (corta a
  mitad de palabra). No se probó un presupuesto mayor.
- **CCS dominado por novedad** (α=0.8, λ=2) y acotado ~0.5 por construcción; su
  interpretación absoluta es floja (por eso es exploratorio).
- **Sin intervalos de confianza.** Las diferencias chicas (soft_ce vs ce_uniform en
  top1_known) están dentro del ruido; la conclusión fuerte es agregación >> weighting y
  SFT >> base, no el orden fino entre las dos mejores.
- **`length_normalize_concept_logprob=true`** quedó activo (marcado experimental) en las 4;
  afecta la geometría de la loss por igual, no invalida la ablation pero conviene revisar.

---

## 9. Próximos pasos (priorizados)

1. **Barato y sin reentrenar — re-decodificar el ganador.** Correr `soft_ce` con
   `max_new_tokens≈12` y `repetition_penalty` más suave para ver si desaparecen los
   tokens corruptos y sube la cobertura exacta. Diagnóstico de una hora.
1.b **Cerrar la métrica de creatividad con juez LLM** (§12): habilitar Anthropic en Model
   Garden y correr la evaluación completa de `soft_ce` con Sonnet (~$5–8). Tubería ya
   implementada (`src/eval/judge_creativity.py`) y validada en piloto.
2. **DPO desde el adapter `soft_ce`** (alto impacto). Preferidos = `known_outputs`
   cortos/canónicos; rechazados = las muestras verbosas/corruptas del propio modelo. Ataca
   de frente brevedad + correctitud. La cuota A100=4 lo habilita en paralelo.
3. **Brevedad por prompt** (requiere reentrenar): prompt del student con límite de
   palabras explícito / few-shot. Es un experimento nuevo con su reentrenamiento.
4. **Higiene de datos:** revisar labels raros del teacher; quizá recortar `known_outputs`
   a ≤2 palabras para alinear target y métrica.
5. **Limpieza de infra:** borrar `checkpoints/` viejos (~500 GiB), hornear el commit SHA
   en la imagen y usar tags inmutables.

---

## 10. Apéndice — reproducción

**Predicciones (GCS):** `gs://llm-craft-bucket/eval_outputs/cu126_<variante>_test/predictions.jsonl`
para `<variante> ∈ {concept_set, softce, concept_set_uniform, ce_uniform, base}`.

**Re-scoring offline (CPU, gratis):**

```bash
OUT=gs://llm-craft-bucket/eval_outputs
uv run python -m src.eval.score_coverage \
  $OUT/cu126_concept_set_test/predictions.jsonl \
  $OUT/cu126_softce_test/predictions.jsonl \
  $OUT/cu126_concept_set_uniform_test/predictions.jsonl \
  $OUT/cu126_ce_uniform_test/predictions.jsonl \
  $OUT/cu126_base_test/predictions.jsonl \
  --labels concept_set,soft_ce,concept_set_uniform,ce_uniform,base \
  --semantic_threshold 0.75
```

**Re-generar una eval (GPU en Vertex, A100 80GB):**

```bash
GCS=/gcs/llm-craft-bucket
uv run --group vertex python -m src.sft.vertex_submit \
  --run-name eval-softce --module src.eval.run_sft_eval \
  --machine-type a2-ultragpu-1g --accelerator-type NVIDIA_A100_80GB --accelerator-count 1 \
  --boot-disk-gb 100 --no-wait \
  -- --run_dir $GCS/runs/2026-07-02_2300_qwen3_4b_thinking_10k_softce \
  --output_dir gs://llm-craft-bucket/eval_outputs/cu126_softce_test \
  --eval_file $GCS/datasets/final-10k/test.jsonl \
  --num_samples 4 --max_new_tokens 8 --temperature 0.6 --top_p 0.9 --top_k 40 \
  --repetition_penalty 1.15 --no_repeat_ngram_size 3 --max_concept_words 3
```

El baseline base agrega `--no_adapter --close_think_prompt true --strip_think true`.

---

## 11. Glosario de métricas: qué mide cada una, pros y contras

Este proyecto tiene **dos familias de métricas** que responden preguntas distintas:
cobertura ("¿acertó a la respuesta esperada?") y creatividad ("¿es buena y nueva la
respuesta?"). Ninguna sola alcanza; se leen juntas. Abajo, cada métrica con su
definición operativa, sus pros y sus contras.

### 11.1 Cobertura exacta — `top1/any@k` contra `known_outputs` / `canonical`

- **Qué mide.** Fracción de recetas donde la muestra coincide *string a string*
  (normalizada: minúsculas + espacios) con alguna salida conocida (`known`) o con la
  principal (`canonical`). `top1` = la primera muestra; `any@k` = alguna de las 4.
- **Pros.** Objetiva, determinista, gratis, reproducible al 100%, sin modelo de por
  medio. Ideal para **elegir entre variantes** (fue la métrica decisoria de la ablation).
- **Contras.** *Brutalmente estricta*: castiga sinónimos (`fax machine` ≠ `faxmachine`),
  verbosidad (`lightning rod system` ≠ `lightning`) y respuestas correctas fuera de la
  lista del teacher (`calcium phosphate`, mejor que el label). Subestima el desempeño
  real (acá ~7% exacto vs ~80% de validez según el juez, §12). Depende de la calidad de
  `known_outputs`.

### 11.2 Cobertura semántica — coseno ≥ 0.75 (MiniLM)

- **Qué mide.** Igual que la exacta, pero una muestra "acierta" si su embedding tiene
  coseno ≥ umbral con algún `known_output`. Rescata sinónimos y paráfrasis.
- **Pros.** Mucho más robusta que la exacta a la forma; barata (CPU) y determinista.
  Buena para estimar "¿está cerca de una respuesta buena?".
- **Contras.** El umbral es arbitrario (0.75) y mueve el resultado; los embeddings de
  frase confunden *relacionado* con *correcto* (`fire` y `firefighter` están cerca sin
  ser lo mismo). No verifica correctitud lógica, solo cercanía semántica.

### 11.3 Verbosidad — media de palabras del top-1 y fracción ≤2 palabras

- **Qué mide.** Cuán corto responde el modelo. La tarea premia conceptos de ≤2 palabras.
- **Pros.** Diagnóstico directo del problema #1 (sobre-especificación). Trivial de
  calcular, muy interpretable.
- **Contras.** No es calidad: una respuesta corta puede ser incorrecta y una larga,
  correcta. Es un *proxy de forma*, se lee junto a cobertura, no sola. Además el
  post-proceso `max_concept_words=3` la recorta artificialmente.

### 11.4 CCS y sus componentes — **proxy geométrico de embeddings** (§5.4)

`C(x) = α·mean(qᵏ·n) + (1−α)·d`, con α=0.8, λ=2. Los tres componentes son **distancias
coseno con `all-MiniLM-L6-v2`**, sin ningún juez:

- **plausibility (`q`)** = 1 − dist(muestra, *centroide de los `known_outputs`*)/2.
  "¿Qué tan cerca está la muestra del promedio de las respuestas buenas?"
- **novelty (`n`)** — en estas corridas, `novelty_method=input_distance`: distancia coseno
  media de la muestra a `input_a` y `input_b`, ÷2. "¿Qué tan lejos del *input* está?"
  (NO mide novedad respecto al dataset).
- **diversity (`d`)** = distancia coseno media entre las 4 muestras. "¿Qué tan variadas
  entre sí son?"
- **Pros.** Barato, sin llamadas a LLM; da una señal continua de forma/variedad; útil como
  chequeo de cordura y para análisis exploratorio.
- **Contras (por qué quedó como secundaria).** (1) **No verifica correctitud**: premió al
  base por "novedad" cuando escupía `"Hmm the user"`. (2) `novelty` mide distancia al
  *input*, que no es lo que entendemos por creatividad. (3) La ponderación (α=0.8, λ=2) y
  el rango (~0.5 por construcción) hacen su valor absoluto poco interpretable. (4)
  **Contradijo a cobertura** y la banda entre variantes fue ruido. ⇒ Sirve como
  descriptor, no como veredicto de creatividad.

### 11.5 Juez LLM — **Validez** y **Creatividad** (§12, la métrica "de verdad")

- **Qué mide.** Un LLM fuerte juzga, sin ver las respuestas del teacher, si la salida es
  un resultado *plausible* de combinar A+B. Se cruza con una comprobación determinista de
  "¿está en el dataset?" para separar dos tasas:
  - **Validez** = fracción correcta (esté o no en el dataset).
  - **Creatividad** = fracción correcta **Y** nueva (fuera de `known_outputs`) = descubrimiento.
- **Pros.** Es lo más cercano a un juicio humano de la tarea; acredita respuestas
  correctas fuera de la lista del teacher (donde la cobertura es ciega); separa "correcto"
  de "creativo" en dos ejes legibles; puede detectar que el alumno *superó* al teacher.
- **Contras.** Cuesta dinero y tiempo (llamadas API); es *no determinista* (aunque temp=0
  ayuda); su calidad depende del prompt/rúbrica y del modelo juez (debe ser **más fuerte
  que el teacher** para ser fiable); puede tener sesgos (lenidad con la verbosidad — por
  eso brevedad sigue siendo eje aparte). Se valida a mano sobre una muestra antes de
  confiar en la corrida completa.

### 11.6 Cómo leerlas juntas (resumen)

| Pregunta | Métrica | Confianza |
|---|---|---|
| ¿Qué variante de loss elijo? | Cobertura exacta/semántica | alta (decisoria) |
| ¿Está cerca de una buena respuesta? | Cobertura semántica | media |
| ¿Responde en el formato correcto (corto)? | Verbosidad | alta (para forma) |
| ¿Es correcta de verdad? | **Validez (juez)** | alta |
| ¿Es correcta **y** nueva? | **Creatividad (juez)** | alta |
| Descriptor barato de forma/variedad | CCS embeddings | baja (exploratoria) |

---

## 12. Creatividad con juez LLM (diseño, decisiones y metodología)

La cobertura es ciega a la respuesta *correcta-pero-fuera-de-lista*. Cuando el alumno
produce algo plausible que el teacher nunca listó (p. ej. `element+skull → calcium
phosphate`, mejor que el label `bone`), la cobertura la puntúa 0. Esta métrica cierra ese
hueco con un **juez LLM fuerte** (más fuerte que el teacher Gemini 2.5 Flash).

### 12.1 Definición de creatividad (Boden): valor × novedad

Un resultado es creativo si es **a la vez** *valioso/plausible* (un resultado sensato de
combinar A+B) **y** *novedoso* (no obvio, no un eco del input, no la única respuesta
trivial). La clave — y el error que hundió al CCS de embeddings — es que **la novedad solo
cuenta si además es válida**: "fuera del dataset" a secas es ambiguo (puede ser un
descubrimiento o simplemente estar mal). Por eso se **condiciona** la novedad a la validez.

### 12.2 Dos ejes independientes

| Eje | Fuente | Definición |
|---|---|---|
| **¿plausible?** | juez Claude/Gemini, **sin referencia** | "¿Es `<salida>` un resultado sensato de combinar A+B?" El juez **no ve** los `known_outputs`, así que no puede solo copiar coincidencias ni penalizar una respuesta válida no listada. |
| **¿en el dataset?** | código determinista | ¿coincide con algún `known_output` (exacto normalizado **o** coseno ≥ 0.75)? |

Cubetas por salida y tasas titulares (sobre N recetas, para top-1 y any@k):

```
invalid      = no plausible                    -> mal / malformado
valid-known  = plausible Y en-dataset          -> correcto, pero no nuevo
valid-novel  = plausible Y fuera-de-dataset    -> DESCUBRIMIENTO (correcto + nuevo)

Validez    = (valid-known + valid-novel) / N   -> con qué frecuencia acierta
Creatividad =  valid-novel / N                 -> con qué frecuencia acierta Y es nuevo
```

### 12.3 Decisiones de diseño (y por qué)

1. **Reportar Validez y Creatividad por separado** (no colapsar en un número). No se pierde
   información: se ve el trade-off correcto-vs-creativo explícito.
2. **Validez sin referencia.** El juez no ve `known_outputs`, para no limitarse a sellar
   coincidencias ni castigar respuestas válidas fuera de lista — esto es lo que permite
   acreditar cuando el alumno supera al teacher. La pertenencia al dataset se decide aparte
   en código.
3. **Juez más fuerte que el teacher.** Un juez tan bueno como el teacher (Gemini 2.5 Flash)
   no podría reconocer cuándo el alumno lo superó. Primera elección: **Claude Sonnet 4.5**
   (proveedor independiente). **Bloqueado por facturación** (ver §12.4). Elección final:
   **Gemini 2.5 Pro** vía **batch** (más fuerte que Flash, ~50% más barato que realtime).
4. **Se juzga solo al ganador `soft_ce`.** Juzgar al base gastaría ~1263 llamadas para
   confirmar ~0 (produce meta-texto). El base queda como piso conocido.
5. **Se juzgan las 4 muestras por receta** en una llamada → top-1 y any@k gratis; se
   deduplican las repetidas (colapso de modo) para no pagar de más.
6. **Brevedad sigue como eje aparte.** El juez es (correctamente) *lene con la verbosidad*
   — plausibilidad ≠ concisión. La brevedad se mide con el conteo de palabras (§11.3).

### 12.4 Infraestructura y hallazgos de acceso (por qué Gemini y no Claude)

- **Claude no está habilitado en el Model Garden de Vertex de este proyecto**: los modelos
  Anthropic devuelven 404 en todas las regiones probadas (`us-central1`, `us-east5`,
  `europe-west1`, `us-east1`) con "your project does not have access".
- **Habilitar Anthropic falla por facturación.** Al intentar habilitarlo en Model Garden:
  *"Upgrade to a full Billing Account to purchase on Marketplace — This product cannot be
  purchased using a credit-based account."* La cuenta actual (créditos) no puede comprar
  productos de Marketplace ⇒ Sonnet queda descartado sin una cuenta de facturación con
  medio de pago registrado.
- La **API de Batches de Anthropic (50% descuento) tampoco existe en el cliente Vertex** —
  el SDK lanza `"The Batch API is not supported in the Vertex client yet"`.
- **Decisión: Gemini 2.5 Pro por `google-genai`, en modo batch.** Es más fuerte que el
  teacher (2.5 Flash), funciona hoy sin habilitaciones, y el **batch de Vertex** (job
  `client.batches.create`, mismo patrón que el enriquecimiento del teacher) da ~50% de
  descuento. El código soporta ambos jueces; el batch es Gemini-only (Claude no lo permite
  en Vertex).

### 12.5 Resultados (completo, 1263 recetas, `soft_ce`, juez Gemini 2.5 Pro en batch)

| Métrica | Juez LLM | Cobertura exacta (referencia) | Cobertura semántica |
|---|---|---|---|
| **Validez** top-1 (correcto) | **0.833** | 0.073 | 0.337 |
| **Validez** any@k | **0.968** | 0.143 | 0.511 |
| **Creatividad** top-1 (correcto **y** nuevo) | **0.536** | — | — |
| **Creatividad** any@k | **0.769** | — | — |

Cubetas del top-1: **inválido 16.7%**, **válido-conocido 29.7%**, **válido-nuevo
(descubrimiento) 53.6%**.

**Lectura.**

1. **El modelo acierta el 83% de las veces** (top-1) según un juez fuerte — vs 7% de
   exact-match y 34% de cobertura semántica. El exact-match subcontaba ~11× por
   verbosidad, sinónimos y respuestas correctas fuera de la lista del teacher.
2. **Más de la mitad de los top-1 son descubrimientos** (correctos **y** fuera del
   dataset): 53.6%. El alumno no memoriza la lista del teacher; produce composiciones
   válidas nuevas (`castle wall+castle wall → castle fortress complex`,
   `mold+smoke → penicillin smoke ring`).
3. **Con 4 muestras, casi siempre hay una válida** (any@k 96.8%) y en el 77% hay al menos
   un descubrimiento válido. Esto refuerza el valor de muestrear K y de un futuro DPO que
   empuje la buena muestra al top-1.
4. **El 16.7% inválido** es el techo de mejora inmediato: son los tokens corruptos y
   fragmentos cortados (`'owl garden a'`, comillas sueltas) que el juez detecta bien —
   atacables con decoding (más tokens, menos `repetition_penalty`) y/o DPO.

El juez (Gemini 2.5 Pro, batch, temp=0) es más fuerte que el teacher (2.5 Flash) y
juzga **sin ver** los `known_outputs`, así que acredita respuestas correctas fuera de la
lista — justo donde la cobertura es ciega. Números en
`gs://llm-craft-bucket/eval_outputs/judge_softce_gemini/{summary.json,judgments.jsonl}`.

> ✅ **Corrida completa hecha.** Costo real: batch de 1263 recetas con Gemini 2.5 Pro,
> dentro del presupuesto (< $20). Antecedente: el piloto n=20 dio 0.80/0.95 validez y
> 0.50/0.70 creatividad — consistente con el completo.

**Limitación del juez.** Es (correctamente) lene con la verbosidad: acepta como plausible
`castle fortress complex` aunque la tarea premie el concepto corto. Por eso **Validez alta
no contradice cobertura exacta baja** — miden cosas distintas (correcto vs correcto-y-en-
formato-canónico). La brevedad se sigue leyendo con §11.3.

### 12.6 Reproducción del juez

```bash
# Recomendado: Gemini 2.5 Pro en BATCH (~50% más barato). Sube requests a GCS, lanza un
# BatchPredictionJob de Vertex, hace polling, baja y parsea los resultados.
uv run --group vertex python -m src.eval.judge_creativity \
  gs://llm-craft-bucket/eval_outputs/cu126_softce_test/predictions.jsonl \
  --model gemini-2.5-pro --mode batch \
  --gcs_staging gs://llm-craft-bucket/judge_batch --poll_seconds 30 \
  --semantic_threshold 0.75 \
  --output_dir gs://llm-craft-bucket/eval_outputs/judge_softce_gemini

# Realtime (una llamada por receta, concurrente) — útil para pilotos rápidos.
uv run --group vertex python -m src.eval.judge_creativity \
  gs://llm-craft-bucket/eval_outputs/cu126_softce_test/predictions.jsonl \
  --model gemini-2.5-pro --mode realtime --concurrency 8 --max_examples 20 \
  --semantic_threshold 0.75

# Sonnet (si algún día se habilita Anthropic con cuenta de pago): solo realtime, us-east5 auto.
#   --model claude-sonnet-4-5@20250929 --mode realtime
```
