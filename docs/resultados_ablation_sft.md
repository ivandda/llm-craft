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
- **El CCS (nuestra métrica de investigación) contradijo a cobertura** y por eso queda
  como secundario/exploratorio, no decisorio.

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
