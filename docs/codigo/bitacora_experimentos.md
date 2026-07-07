# Bitácora de experimentos — SFT en Vertex AI

Registro de lo que se entrenó y evaluó hasta ahora en GCP, reconstruido a partir de
los `args` de los Custom Jobs, los `run_dir` en el bucket y las métricas guardadas.
El objetivo es que el estado del proyecto quede claro y no haya que reconstruirlo
de memoria. Última actualización: 2026-07-07.

> **Actualización 2026-07-07: la ablation ya se evaluó río abajo.** Las 4 celdas de
> loss + el baseline base se corrieron sobre el test completo (1263 recetas). Ganó
> **`soft_ce` (dataset × expected_logprob)** por cobertura. Detalle en
> [Resultados de la ablation](#resultados-de-la-ablation-evaluación-generativa-completa-2026-07-07).

## Infraestructura

- Proyecto GCP: `nlp2026-498021` (número `486944883203`).
- Bucket: `gs://llm-craft-bucket` (datos en `datasets/`, corridas en `runs/`, evals en `eval_outputs/`).
- Región: `us-central1`. Hardware: 1× **NVIDIA L4** (`g2-standard-8`).
- Modelo base: **`Qwen/Qwen3-4B-Thinking-2507`**.
- Imagen de training en Artifact Registry (`llm-craft-registry/llm-craft-sft`), lanzada con `src/sft/vertex_submit.py`.

## Configuración común de entrenamiento

Todas las corridas 4B parten de `configs/sft/qwen3_4b_thinking_10k_example.yaml` y
solo difieren en los ejes de la loss (pasados como override por CLI):

- QLoRA 4-bit (nf4), LoRA `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=auto`.
- 3 épocas → **30.495 steps**, `lr=2e-4`, `bf16`, gradient checkpointing.
- `prompt_format: qwen_chat`, `max_seq_length=512`, `seed=42`.
- `length_normalize_concept_logprob: true` (ver advertencias).
- Dataset: `datasets/final-10k` (`train.jsonl` + `dev.jsonl`, `max_dev_examples=1291`).

Aclaraciones que suelen generar confusión:

- **Base = versión Thinking**: las 4 usan `Qwen/Qwen3-4B-Thinking-2507` como
  checkpoint base, pero la supervisión es **solo sobre el concepto final** (sin
  `<think>` ni rationale en el target). Por eso responden directo en inferencia:
  "versión thinking" es el checkpoint base, no el comportamiento entrenado.
- **Ninguna entrenó con rationale**: `rationale_loss_weight = 0.0` en las cuatro
  (las dos corridas previas al soporte de rationale ni siquiera tienen el campo; el
  default del código es `0.0`). La única corrida rationale-first quedó cancelada.

### Thinking vs. rationale (y por qué importa acá)

Son dos cosas distintas que es fácil confundir porque ambas son "texto extra"
alrededor del concepto:

- **Thinking (razonamiento del modelo base).** `Qwen3-4B-Thinking-2507` es un modelo
  de razonamiento: por defecto genera una cadena de pensamiento entre `<think>…</think>`
  antes de la respuesta. Es un **comportamiento del checkpoint base**, no viene de
  nuestros datos, es largo y libre, y **no lo supervisamos**. En nuestra tarea no lo
  queremos en la salida (pedimos un concepto de ≤2 palabras).
- **Rationale (dato nuestro).** Es una explicación breve que el *teacher* (Gemini)
  generó para cada `candidate_output` (p.ej. *"fire heats water, producing steam"*).
  Vive en el dataset (`candidate_outputs[].rationale`) y **se puede enseñar
  explícitamente** como término auxiliar de loss (`rationale_loss_weight`). Es corto,
  estructurado, opcional, y ninguno de los 4 modelos lo usó.

Resumen: *thinking* = razonamiento libre del modelo base (no supervisado, formato
largo); *rationale* = texto corto y curado de nuestros datos que podríamos entrenar.
Son ejes independientes: usar el checkpoint "Thinking" **no** implica "entrenado para
razonar/justificar".

Por qué importa para este trabajo:

1. **Evaluación justa.** Los 4 modelos SFT responden directo (se los supervisó solo
   sobre el concepto). Si el modelo base razona (thinking) y los SFT no, hay que
   comparar únicamente la **respuesta final** — por eso `run_sft_eval.py` agrega
   `--no_adapter` para el baseline base y `--enable_thinking`/`--strip_think` para
   aislar/quitar el `<think>`.
2. **Métrica y brevedad.** La tarea apunta a ≤2 palabras; si thinking o rationale se
   filtran en la salida, rompen el exact-match de cobertura y la meta de concisión.
3. **Diseño experimental.** "Entrenar con rationale" es una palanca futura (la variante
   rationale-first, hoy cancelada), no algo que ya esté en estos 4 modelos.

## La matriz de losses (2×2)

La familia de losses se parametriza por `candidate_weighting × candidate_aggregation`.
Las cuatro celdas se entrenaron con datos idénticos, lo que la vuelve una ablation
controlada del lado de entrenamiento.

| Run dir | Loss (`weighting × aggregation`) | Alias | Estado | Eval de creatividad |
|---|---|---|---|---|
| `2026-07-01_1950_qwen3_4b_thinking_10k` | dataset × logsumexp | `concept_set` (default) | ✅ | ✅ (única evaluada) |
| `2026-07-02_2300_qwen3_4b_thinking_10k_softce` | dataset × expected_logprob | `soft_ce` | ✅ | ❌ |
| `2026-07-03_2250_qwen3_4b_thinking_10k_concept_set_uniform` | uniform × logsumexp | `concept_set_uniform` | ✅ | ❌ |
| `2026-07-03_2255_qwen3_4b_thinking_10k_ce_uniform` | uniform × expected_logprob | `ce` | ✅ | ❌ |

Corridas relacionadas que **no** forman parte de la matriz:

- `2026-06-29_1854_qwen3_4b_thinking_10k`: primera prueba, previa a la sugerencia
  de Luciano de hacer el update por receta. **Superada**, no cuenta como celda.
- `2026-07-04_1748_qwen3_4b_rationale_first`: variante con rationale-first,
  **CANCELADA** (nunca terminó).
- `2026-07-03_2105_..._concept_set_uniform`: run dir **vacío** (reintento fallido
  del `concept_set_uniform`; la corrida buena es la de las `2250`).

## Resultado de entrenamiento (dev loss)

Las cuatro corridas convergieron de forma estable (train loss de ~13–17 a ~1.0–1.3,
dev loss monótonamente decreciente hasta el final).

| Variante | dev loss final |
|---|---|
| `concept_set` (dataset × logsumexp) | 1.625 |
| `concept_set_uniform` (uniform × logsumexp) | 1.734 |
| `soft_ce` (dataset × expected_logprob) | 2.054 |
| `ce` (uniform × expected_logprob) | 2.299 |

⚠️ **Estos números NO ordenan a los modelos entre sí.** `logsumexp` y
`expected_logprob` son objetivos distintos y en escalas distintas: el log-sum-exp
sobre un conjunto de candidatos es matemáticamente menor que promediar la NLL por
candidato. Por eso `concept_set` tiene menor loss *por construcción*, no por ser
mejor modelo. La **única** comparación justa es la evaluación generativa río abajo,
que todavía falta correr en 3 de las 4 celdas. La única comparación válida acá es
*weighting vs uniform dentro de la misma agregación*: `dataset` da menor loss que
`uniform` en ambas filas (1.625<1.734 y 2.054<2.299), una señal débil a favor de
ponderar, pero es LM loss, no creatividad.

## Evaluación hecha hasta ahora

Todas las evals de creatividad se corrieron sobre **un solo modelo**, el
`concept_set` (dataset × logsumexp) de `2026-07-01_1950`, con
`src/eval/run_sft_eval.py`. Cuatro carpetas en `eval_outputs/`:

| Carpeta | Alcance | CCS | Notas |
|---|---|---|---|
| `2026-07-01_1950_qwen3_4b_thinking_10k_full_eval` | test completo (1263 recetas), ~40 min | 0.323 | plaus 0.779 · novelty 0.304 · diversity 0.886 |
| `qwen3_4b_l4_eval_100_input_distance` | 100 recetas | 0.304 | tuning de settings de eval |
| `qwen3_4b_l4_eval_100_input_distance_tight` | 100 recetas | 0.313 | idem |
| `qwen3_4b_l4_eval_100_input_distance_tight_v2` | 100 recetas | 0.321 | idem |

Config de eval usada: `num_samples=4`, `max_new_tokens=8`, `temperature=0.6`,
`top_p=0.9`, `top_k=40`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`,
`alpha=0.8`, `lambda=2.0`, `novelty_method=input_distance`, embeddings
`sentence-transformers/all-MiniLM-L6-v2`.

Fórmula operacional (CCS): `C(x) = α·mean(q^λ·n) + (1−α)·d`, con `q`=plausibilidad,
`n`=novedad (distancia del output a los dos inputs / 2), `d`=diversidad. Con `α=0.8`
y `λ=2`, el CCS queda dominado por el término de novedad, que es bajo (~0.30) y está
acotado ~0.5 por construcción.

### Lectura cualitativa

Los outputs son plausibles pero conservadores y repetitivos: tienden a concatenar
los inputs (`hut+steam → "steamboat cabin hut"`), a repetir la misma muestra
(`cream+ice → "ice cream popper"` ×3) o a derivar (`garden+owl → "nightingale
songbird shelter"`). Esto explica la novedad baja (los outputs quedan cerca de los
inputs) y la plausibilidad decente.

### Baseline base (sin SFT)

El job `2026-07-01 qwen3-4b-base-dev-eval` corrió `src/sft/eval_base.py`, que solo
calcula la **dev LM-loss** del modelo base (no genera, no calcula creatividad) y solo
la imprime en logs (sin archivo). Ese número ya no es recuperable desde los logs.
➡️ **No existe todavía un baseline de creatividad del modelo base**, así que hoy no
se puede afirmar que el SFT mejore la creatividad respecto de Qwen sin entrenar.

> Aclaración de nombres: el job `2026-07-06 qwen3-4b-eval-full-l4-base`, pese al
> "base" en el nombre, re-evaluó el *adapter* `2026-07-01`, no el modelo base.

## Resultados de la ablation (evaluación generativa completa, 2026-07-07)

Las **4 celdas de loss** + el **baseline base** (`--no_adapter`) se evaluaron sobre el
**test completo (1263 recetas)** con `src/eval/run_sft_eval.py` y luego se puntuaron
offline con `src/eval/score_coverage.py`. Config de decoding idéntica a la corrida
validada del 07-01 (`num_samples=4`, `max_new_tokens=8`, `temperature=0.6`,
`top_p=0.9`, `top_k=40`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`,
`max_concept_words=3`). Predicciones en `gs://llm-craft-bucket/eval_outputs/cu126_*_test/`.

**Infra:** imagen reconstruida con **torch 2.12.1+cu126** (la de entrenamiento era
cu130, incompatible con el driver CUDA 12.2 de los nodos T4/A100 → caía a CPU). Las 5
evals corrieron en **A100 80GB, todas en bf16** (misma precisión ⇒ comparación limpia;
se descartó T4 porque solo soporta fp16). Ver
[Crisis de GPU](#crisis-de-gpu-imagen-cuda-12-2026-07-0607) más abajo.

### Cobertura (métrica primaria) — coincidencia con `known_outputs`

| Modelo (weighting × aggregation) | top1_known | any@k_known | top1_canon | any@k_canon | ≤2 palabras |
|---|---|---|---|---|---|
| **`soft_ce`** (dataset × expected_logprob) | 0.073 | **0.143** | **0.042** | **0.073** | **40.9%** |
| `ce_uniform` (uniform × expected_logprob) | **0.075** | 0.134 | 0.038 | 0.064 | 33.7% |
| `concept_set` (dataset × logsumexp) | 0.040 | 0.077 | 0.026 | 0.041 | 24.0% |
| `concept_set_uniform` (uniform × logsumexp) | 0.038 | 0.075 | 0.025 | 0.041 | 18.2% |
| **base** (sin SFT, `--close_think_prompt`) | **0.000** | **0.000** | **0.000** | **0.000** | 1.3% |

Cobertura semántica (coseno ≥0.75 a un `known_output`, MiniLM), como robustez:

| Modelo | top1_sem | any@k_sem |
|---|---|---|
| `soft_ce` | **0.337** | **0.511** |
| `ce_uniform` | 0.290 | 0.476 |
| `concept_set` | 0.274 | 0.410 |
| `concept_set_uniform` | 0.259 | 0.394 |
| base | 0.000 | 0.000 |

### CCS (métrica secundaria / exploratoria)

| Modelo | CCS | plaus | novelty | diversity |
|---|---|---|---|---|
| `concept_set` | 0.324 | 0.780 | 0.304 | 0.887 |
| `concept_set_uniform` | 0.321 | 0.776 | 0.306 | 0.877 |
| `soft_ce` | 0.318 | 0.786 | 0.303 | 0.850 |
| `ce_uniform` | 0.313 | 0.781 | 0.301 | 0.841 |
| base | 0.284 | 0.600 | 0.409 | 0.829 |

### Conclusiones

1. **El SFT es necesario y efectivo.** El base saca **0.0% de cobertura** en todas las
   métricas (exacta, canónica y semántica), incluso con el mejor trato posible del
   thinking (`--close_think_prompt` + `--strip_think`). Cualitativamente el base no
   produce conceptos sino *fragmentos de razonamiento*: `cream+ice → "Hmm the user"`,
   `energy+tree → "We are combining"`. Los SFT sí producen conceptos on-task
   (`energy+tree → "photosynthesis"` exacto; `hut+steam → "sauna hut"`).
2. **Ganador: `soft_ce` (dataset × expected_logprob).** Lidera en 6 de 7 métricas de
   cobertura (todas salvo top1_known, donde `ce_uniform` empata a 0.2 pp, dentro del
   ruido) y es el **menos verboso** (40.9% de respuestas ≤2 palabras, alineado con la
   preferencia de outputs cortos).
3. **El eje que manda es la *agregación*, no el *weighting*.** `expected_logprob`
   (soft_ce, ce_uniform ≈ 7.5% top1) **duplica** a `logsumexp` (concept_set,
   cs_uniform ≈ 4%). Dentro de cada agregación, `dataset` vs `uniform` apenas se
   distingue. Interpretación: promediar la NLL por candidato (soft-CE) enseña a
   generar la respuesta *modal*; el log-sum-exp reparte masa sobre el conjunto y
   diluye el top-1.
4. **CCS contradice a cobertura y por eso queda como secundaria.** El CCS rankea
   `concept_set` **primero** (0.324) — exactamente al revés que cobertura — porque
   premia diversidad/novedad (concept_set diverge más de los inputs) sin verificar si
   la respuesta es *correcta*. Esto confirma la decisión de usar cobertura como métrica
   primaria y tratar el CCS como exploratorio. La diferencia de CCS entre las 4 celdas
   (0.313–0.324) es además ruido; no sirve para elegir.

➡️ **Decisión: adoptar `soft_ce` (dataset × expected_logprob) como la loss ganadora del
SFT.** Las cifras absolutas son bajas (7% top1 exacto) porque el target admite muchos
sinónimos no listados; la cobertura semántica (34% top1 / 51% any@k) y los ejemplos
cualitativos muestran que las respuestas son razonables. Punto de partida sólido para
un eventual DPO.

## Procedencia y reproducibilidad

Cada `run_dir` guarda su procedencia en GCS (persistente, independiente de cualquier laptop):

- `command.txt`: el CLI exacto que se ejecutó.
- `config.yaml`: la config **resuelta** efectiva (todos los flags tras aplicar overrides).
- `data_fingerprint.json`: SHA-256 de `train`/`dev` + versiones de librerías.
- `git_info.json`: commit (ver limitación abajo).

Confirmado con evidencia:

- **Datos idénticos** en las 4 corridas: train `3399f18…` (10.165 líneas), dev
  `cd6c911…` (1.291 líneas). Mismo SHA-256, no solo mismo nombre de archivo.
- **Entorno pinneado**: `transformers 5.12.1`, `torch 2.12.1+cu130`, `peft 0.19.1`,
  `accelerate 1.14.0`, `bitsandbytes 0.49.2`, Python 3.12.13.
- **Config idéntica salvo los ejes de loss**; `length_normalize_concept_logprob: true`
  y `rationale_loss_weight: 0.0` en las 4.

Limitación y su resolución:

- `git_info.json` da `available: false` (git no estaba en el contenedor) y la imagen
  usaba tag `:latest`, así que **no hay SHA de commit por corrida**.
- El único cambio de código relevante en la ventana (07-01 → 07-03) fue `bf3d166`
  (07-02), que refactorizó `losses.py`/`trainer.py`. **Verificado en el diff**: el
  cálculo de la loss de concepto es idéntico (se renombró `causal_concept_logprobs`
  → `causal_masked_logprobs` con la misma matemática; el término de rationale está
  gateado por `rationale_loss_weight > 0`, y las 4 corridas usan `0.0`). ⇒ Las 4
  celdas computaron la **misma** loss de concepto y difieren **solo** en el eje
  `weighting × aggregation`. La ablation es limpia.
- **Recomendación de higiene:** hornear el commit SHA en la imagen (o registrarlo
  desde el trainer) y usar tags inmutables (`:$(git rev-parse --short HEAD)`, ya
  soportado por `cloudbuild.yaml`), para no tener que reconstruir esto a mano nunca más.

## Prompt del student (train/eval) vs. prompt del teacher

Son dos prompts distintos con roles distintos; no tienen por qué coincidir.

- **Prompt del teacher (complejo).** Vive en `enrich_teacher.py`. Se le manda a Gemini
  para *generar los datos*: dado un par, producir hasta 5 outputs plausibles,
  rankearlos, escribir rationales, rechazar los malos. Da forma al **contenido** del
  dataset (cuáles son las etiquetas), no al formato con que se prompea al student.
- **Prompt del student (simple).** `render_user_prompt` en `collator.py`. Es lo que el
  modelo chico (Qwen) ve en entrenamiento y evaluación. Da forma al **formato** de la
  tarea. Texto exacto:
  - system: `You combine two concepts into one resulting concept.`
  - user:
    ```
    Given two concepts, combine them into one resulting concept.

    Concept A: {input_a}
    Concept B: {input_b}

    Return only the resulting concept.
    ```

Regla dura: el prompt de **train y eval del student deben ser idénticos**. Las 4
corridas se entrenaron con el prompt simple, así que la evaluación **debe** usar el
mismo. Cambiar el prompt del student no es un ajuste de config: **requiere reentrenar**.
Rediseñarlo (instrucciones más fuertes, few-shot, límite de palabras explícito) es un
experimento futuro con su propio reentrenamiento, no una palanca de esta ronda. El
prompt del teacher es independiente y puede seguir siendo complejo.

## Baseline base: manejo del thinking (smoke 2026-07-06)

`Qwen3-4B-Thinking-2507` es un modelo de razonamiento cuyo chat template **inyecta
siempre `<think>`** al final del prompt de generación (`add_generation_prompt=True`) e
**ignora `enable_thinking=False`** (verificado: el prompt renderizado es idéntico con y
sin la flag). Los modelos SFT aprendieron a saltarse ese bloque y responder directo; el
modelo base **sí** razona.

Hallazgos de las smoke runs (`--no_adapter`, 20 recetas del test):

- `--enable_thinking false` → no sirve (el template lo ignora).
- `--strip_think` con `max_new_tokens=200` → el base razona pero **no cierra `</think>`
  dentro del presupuesto**; salen fragmentos de razonamiento (p.ej. *"Okay the user
  wants"*), no un concepto.
- En prueba: `--strip_think` con `max_new_tokens=512` para ver si el base termina de
  razonar y se recupera el concepto tras `</think>`.

Implicancia de costo: si el base necesita ~512 tokens y los SFT responden en ~3, la
comparación de las 5 se hará sobre un **subconjunto fijo** (~200–300 recetas), y
opcionalmente el ganador SFT sobre el test completo.

## Qué falta y advertencias

1. ~~**La ablation nunca se evaluó río abajo.**~~ ✅ **Resuelto (2026-07-07):** las 4
   celdas se evaluaron sobre el test completo; ganó `soft_ce`. Ver
   [Resultados de la ablation](#resultados-de-la-ablation-evaluación-generativa-completa-2026-07-07).
2. ~~**Falta baseline de creatividad del modelo base.**~~ ✅ **Resuelto (2026-07-07):**
   se agregó `--no_adapter` a `run_sft_eval.py` y se corrió el baseline base (0.0% de
   cobertura). El modo sin-adapter ya existe.
3. **No rankear por dev loss** entre losses distintas (ver arriba).
4. **`best_metric` / `best_model_checkpoint` = `None`** en todas las corridas: la
   selección de "mejor adapter" no está atada a una métrica. Fue inocuo (dev loss
   bajó monótonamente, así que best≈final), pero es frágil.
5. **`length_normalize_concept_logprob: true`** quedó activo en todas las corridas
   reales pese a estar marcado como experimental (el default es `false`). Cambia la
   geometría de la loss; conviene decidir si fue intencional.
6. **Higiene de registros:** run dir vacío duplicado (`2105`), job "base" que evaluó
   un adapter. Esta bitácora existe justamente para evitar esa confusión.
7. **Robustez de eval:** `max_new_tokens=8` trunca outputs; muestras repetidas entre
   sí; el tuning de settings de eval se hizo sobre un solo modelo.

## Deuda técnica: checkpoints y staging de eval

**Problema.** Cada `run_dir` pesa ~115 GiB, casi todo en `checkpoints/` (~38 GiB por
checkpoint × 3): `accelerate` guarda el estado completo de optimizer/modelo para
`--resume_from_checkpoint`, aunque en LoRA lo único entrenable sea el adapter (137 MiB).
Con ~4 corridas son **~500 GiB en el bucket (~$10/mes)** de peso muerto: el
entrenamiento ya terminó y los adapters (`best_adapter`/`final_adapter`) se guardan
aparte. Además, `run_sft_eval` **descargaba el `run_dir` entero** al evaluar, lo que
disparó el fallo de disco (100 GB) y ~17 min de startup por job.

**Fix aplicado (2026-07-06).**
- `run_sft_eval` / `download_gcs_prefix` ahora **excluye `checkpoints/`** al stagear el
  `run_dir` (baja de ~115 GiB a ~150 MiB; startup de ~17 min a segundos).
- `vertex_submit` expone `--boot-disk-gb` (default 200) para dar margen.

**Pendiente (a resolver más adelante, NO hacer ahora):**
- **Limpieza:** borrar los `checkpoints/` de las corridas viejas conservando los
  adapters (libera ~500 GiB). Es destructivo: verificar antes que cada `best_adapter`/
  `final_adapter` esté intacto.
- **Entrenamiento futuro:** bajar `save_total_limit` o guardar solo el adapter por
  checkpoint (contra: resume más débil). Aplica recién cuando se vuelva a entrenar.

## Crisis de GPU: imagen CUDA-12 (2026-07-06/07)

**Síntoma.** Al intentar correr las evals, los jobs en T4 y A100 caían a **CPU** (una
eval de 4B tarda horas), mientras que L4 sí usaba GPU pero tenía poca capacidad
(jobs encolados).

**Causa raíz.** El wheel de `torch 2.12.1` por defecto de PyPI trae runtime **CUDA 13**
(`nvidia-*-cu13`), que exige un driver CUDA 13. Los nodos T4/A100 de Vertex en
us-central1 tienen driver **CUDA 12.2** (`found version 12020`), un major por detrás ⇒
sin compatibilidad ⇒ torch no ve la GPU y cae a CPU. Solo los nodos L4 tenían driver
nuevo, de ahí que fuera el único que "funcionaba".

**Fix (2026-07-07).** Se ancló torch al índice **cu126** de PyTorch (solo en Linux; en
macOS sigue el wheel de PyPI para no romper el dev local ni el `uv lock`):

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
minor-version compatibility). Se reconstruyó la imagen y un smoke en A100 80GB confirmó
`Model ready on device cuda:0`. Esto **desbloquea A100 y T4** además de L4.

**Cuota.** Se subió la cuota `CustomModelTrainingA10080GBGPUsPerProjectPerRegion` en
us-central1 de **1 → 4** (via Cloud Quotas API), permitiendo correr evals/trainings en
paralelo. (Quedó una preferencia vieja en `africa-south1` con granted=0, inocua.)

**Robustez de submit.** Se agregó `--no-wait` a `vertex_submit.py`: hace `job.submit()`
(async) en vez de `job.run()` (bloqueante), para que un proceso local que se muera no
sea single-point-of-failure de un batch. ⚠️ **Lección:** al pasar flags extra por
`-- ...`, hacerlo como tokens separados (o array bash `"${GEN[@]}"`), **no** como una
string sin comillas: dos jobs fallaron porque los 9 flags de decoding llegaron como
**un solo token** argv y argparse los rechazó.

## Próximo paso sugerido

> ✅ **El plan de abajo se ejecutó (2026-07-07).** Resultado: ganó `soft_ce`. Ver
> [Resultados de la ablation](#resultados-de-la-ablation-evaluación-generativa-completa-2026-07-07).
> Lo que sigue a partir de acá:
> - **(opcional) DPO** partiendo del adapter `soft_ce`, usando `known_outputs` como
>   preferidos y las muestras que no matchean como rechazados.
> - **Limpieza de `checkpoints/`** (~500 GiB de peso muerto), cuando se decida.
> - **Reentrenar solo si** se quiere probar un prompt de student más fuerte / límite de
>   palabras explícito (requiere reentrenar, ver sección de prompts).

El plan original (ya ejecutado), para referencia:

1. **Métrica primaria = cobertura/accuracy** contra `known_outputs` (top-1 y top-K
   sobre las 4 muestras; exacta normalizada + opcional semántica por embeddings).
   Es una métrica conocida y de bajo riesgo para elegir la loss. `metrics.py` ya la
   implementa, falta cablearla.
2. **CCS = secundaria, no decisoria todavía.** Se computa en la misma pasada (es casi
   gratis: aritmética sobre embeddings), pero es nuestra contribución en investigación
   y todavía se está validando; no se rankea por ella en esta primera vuelta.
3. **Desacoplar generación de scoring.** La parte cara (GPU) es generar y guardar
   `predictions.jsonl`; una vez guardado, cualquier métrica string/embedding se calcula
   offline en CPU, gratis y repetible. Métricas con LLM-judge sí cuestan aparte.
4. **Correr `run_sft_eval.py` sobre las 4 celdas + baseline base** (`--no_adapter`),
   con una config de eval fija, outputs cortos (~8–10 tokens, la tarea es ≤2 palabras),
   en paralelo. Manejar el `<think>` del modelo base (smoke check para decidir si se
   desactiva el thinking o se recorta post-hoc). Costo: pocos dólares, ~1h wall-clock.
5. Consolidar todo en una tabla comparativa (cobertura + CCS) + una tabla de ejemplos
   cualitativos base vs SFT.

Comparación justa (apples-to-apples): los 4 modelos SFT responden **directo** (fueron
supervisados solo sobre el concepto, sin thinking; se verificó en los outputs guardados),
así que el baseline base también se evalúa en modo respuesta-directa.
