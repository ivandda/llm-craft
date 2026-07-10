# llm-craft: Destilación de Creatividad Composicional

Este repositorio contiene las herramientas y el pipeline de datos para el proyecto de destilación y evaluación de creatividad composicional en LLMs pequeños, inspirado en el juego *Infinite Craft*.

---

## Configuración Inicial

Para instalar las dependencias y configurar el entorno virtual utilizando `uv`:

```bash
# Sincronizar e instalar dependencias
uv sync
```

---

## Base de datos local

La app web y los splits `train`/`dev`/`test` del dataset curado `final-10k`
usan Postgres local con Docker. Los datos persisten en `var/postgres-data/`,
fuera de git.

```bash
docker compose up -d postgres
uv run python -m src.data.db_migrate
uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k
```

La URL por defecto es:

```text
postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft
```

La guía completa está en [postgres_dataset_db.md](docs/codigo/postgres_dataset_db.md).
El runbook operativo está en [postgres-local-db-runbook.md](docs/postgres-local-db-runbook.md).

---

## Ejecución del Pipeline completo

Puedes ejecutar todo el pipeline de procesamiento de datos secuencialmente usando el script maestro:

```bash
uv run python -m src.data.run_pipeline
```

Toda la ejecución se puede configurar de forma centralizada en el archivo de configuración:
* **Configuración**: [pipeline_config.yaml](configs/pipeline_config.yaml) (permite controlar fuentes raw, filtros de calidad, proporciones de splits, exclusión de copias de identidad, prompt templates y tamaños de sets de evaluación).

---

## Pasos Individuales del Pipeline

### 1. Normalización de Datos
Carga las distintas fuentes de datos crudos (`datasets/raw/`) y unifica sus formatos en un único archivo de observaciones.
```bash
uv run python -m src.data.normalize
```
* **Qué hace**: Parsea JSONL, CSV y estructuras de Infinite Craft, propaga y unifica emojis case-insensitively, ordena alfabéticamente los ingredientes y escribe en `recipe_observations_v0.jsonl` (Bronze layer).

### 2. Limpieza y Splits (Capa de Plata)
Deduplica observaciones raw, unifica a minúsculas y asigna splits deterministas.
```bash
uv run python -m src.data.clean
```
* **Qué hace**: Convierte todos los conceptos a minúsculas, filtra recetas ruidosas o no comunes, agrupa las recetas duplicadas por par de entrada, identifica conflictos (`is_conflicting_pair`), y asigna cada combinación a `train`/`dev`/`test` usando hashes del par de entrada para evitar filtración de datos (leakage). Genera `recipe_canonical_v0.jsonl`, `clean_metrics.json` y muestras revisables en `quality_reject_samples.jsonl`.

### 3. Exportación de Recetas para Fine-Tuning (SFT) (Capa de Oro)
Prepara datasets mínimos de recetas para entrenamiento supervisado. Los prompts no se guardan en `datasets/processed`; se inyectan en runtime desde los scripts de entrenamiento/evaluación.
```bash
uv run python -m src.data.export_sft
```
* **Qué hace**: Exporta `recipes_train.jsonl`, `recipes_dev.jsonl` y `recipes_test.jsonl`. Cada fila tiene `input_a`, `input_b` y `outputs`, por ejemplo `{"input_a":"fire","input_b":"water","outputs":["steam","mist"]}`.

### 4. Exportación de Conjuntos de Evaluación
Genera los datasets estructurados con respuestas alternativas válidas conocidas para evaluar al estudiante.
```bash
uv run python -m src.data.export_eval
```
* **Qué hace**: Utiliza *Reservoir Sampling* de un solo paso de lectura para extraer muestras aleatorias deterministas o exportar el split completo, según `evaluation_export.sizes`. Cada registro es mínimo y lista las respuestas válidas conocidas (`known_outputs`) asociadas al par.
* **Tamaño completo**: En `evaluation_export.sizes`, usar `all` en lugar de un número para exportar todo el split (`eval_dev_all.jsonl`, `eval_test_all.jsonl`).

### 5. Enriquecimiento con Teacher (manual)
Genera un dataset derivado agrupado con hasta cinco salidas por receta, sin rationales. Conserva las salidas observadas de `recipes_train/dev/test.jsonl` y usa Gemini 2.5 Flash solo para completar alternativas faltantes.

Camino recomendado actual: enriquecimiento estructurado en una sola llamada con Gemini 2.5 Flash. El teacher conserva salidas observadas buenas, agrega alternativas solo si son plausibles, ordena candidatos de mejor a peor receta y escribe rationales breves para todas las salidas aceptadas. Los registros parciales son válidos: no se fuerzan respuestas.

Smoke test real mínimo durante desarrollo:
```bash
uv run python -m src.data.enrich_teacher --splits train --limit 10 --no-resume
```

Prueba de costo/calidad:
```bash
uv run python -m src.data.enrich_teacher --splits train --limit 100 --no-resume
```

El script realtime escribe `datasets/enriched/dataset_04_teacher_ranked_flash_strict_v2/manifest.json` con uso de tokens y costo estimado. Las recetas rechazadas van a `rejected.jsonl`.

Para el dataset completo, usar batch de Vertex AI:
```bash
# 1. Exportar requests locales y el índice de postproceso
uv run python -m src.data.enrich_teacher --mode batch-export --no-resume

# 2. Subir datasets/enriched/dataset_04_teacher_ranked_flash_strict_v2/batch_requests.jsonl a GCS
#    y lanzar el batch. Reemplazar bucket/prefix.
uv run python -m src.data.enrich_teacher \
  --mode batch-submit \
  --batch-input-uri gs://BUCKET/PREFIX/batch_requests.jsonl \
  --batch-output-uri gs://BUCKET/PREFIX/batch_output \
  --batch-display-name llm-craft-teacher-ranked-v2

# 3. Descargar el JSONL generado por Vertex y convertirlo al dataset enriquecido
uv run python -m src.data.enrich_teacher \
  --mode batch-import \
  --batch-output-files /path/to/downloaded/batch_output.jsonl
```

El LLM solo devuelve `keep_recipe`, `reject_reason` y `candidate_outputs`. El código agrega `rank`, `quality_status`, metadata, uso de tokens y costo estimado. Batch tiene descuento aproximado de 50%; el manifest de import escribe costo realtime y costo batch estimado.

### 6. Rationales con Teacher (manual)
Agrega una explicación breve a cada `candidate_output` del dataset multi-salida ya generado. No crea nuevas salidas: valida que el teacher mantenga exactamente los outputs existentes y escribe el resultado en `datasets/enriched/dataset_02_teacher_enriched_multi_output_with_rationale/`.

Smoke test real mínimo:
```bash
uv run python -m src.data.enrich_rationales --splits train --limit 3 --no-resume
```

La generación completa queda como paso manual para controlar costo:
```bash
uv run python -m src.data.enrich_rationales
```

---

## SFT QLoRA

La pipeline de SFT vive aislada en `src/sft/` y entrena modelos causales autoregresivos sobre datasets JSONL con candidatos múltiples por receta. Soporta LoRA/QLoRA y una familia única de losses sobre recetas completas, parametrizada por `candidate_weighting` y `candidate_aggregation`.

La configuración base está en [default.yaml](configs/sft/default.yaml). Actualmente está preparada como smoke test local viable con `sshleifer/tiny-gpt2`, sin 4-bit ni mixed precision, para poder validar el flujo completo incluso sin GPU.

Instalar dependencias:

```bash
uv sync
```

### Estructura de archivos

```text
src/sft/
  __init__.py
  config.py      # Dataclass SFTConfig, carga YAML + overrides CLI y validación.
  dataset.py     # Lectura JSONL, normalización de candidatos y pesos por receta.
  collator.py    # Construye prompts, concept_mask con offsets y aplana candidatos.
  losses.py      # Familia unificada de losses sobre los tokens del concepto final.
  trainer.py     # Loop con accelerate, LoRA/QLoRA, eval, checkpoints y adapters.
  train.py       # Entry point: crea run_dir, guarda metadata y lanza train().
  plotting.py    # Genera plots de train/dev loss al final.
  utils.py       # Seeds, fingerprints, git info, JSON/YAML y manejo de checkpoints.

configs/sft/
  default.yaml   # Defaults del smoke local y parámetros editables.

tests/sft/
  test_config.py
  test_dataset.py
  test_collator.py
  test_losses.py
  test_smoke_train.py
```

La guía técnica extendida está en [sft_qlora_pipeline.md](docs/codigo/sft_qlora_pipeline.md).

## Evaluación generativa y creatividad

El repositorio incluye un evaluador generativo para adapters SFT en `src/eval/run_sft_eval.py`. El script:

* carga un `run_dir` de `runs/sft/`,
* genera `K` respuestas por input,
* y calcula una versión operacional del **Compositional Creativity Score (CCS)** del survey en `Survey_grupo_15.pdf`.

La fórmula implementada sigue:

```text
C(x) = α * mean(q(x, y)^λ * n(x, y)) + (1 - α) * d(Y(x))
```

donde:

* `q(x, y)` se calcula a partir de la **distancia coseno entre la muestra `y` y el embedding promedio de los candidatos conocidos** de esa receta. Se reporta la distancia y, para la fórmula, se usa `1 - distancia / 2` para mantener el score en `[0, 1]`.
* `n(x, y)` se calcula como la **distancia coseno promedio entre el output y los dos inputs de la receta** (`input_a`, `input_b`), dividida por `2` para mantener el score en `[0, 1]`.
* `d(Y(x))` parte de la distancia coseno promedio entre las `K` muestras generadas, y se transforma a score como `1 - distancia / 2`.

Las embeddings para plausibilidad, novedad y diversidad son configurables:

* `--embedding_backend glove`
* `--embedding_backend word2vec`
* `--embedding_backend sentence_embeddings`

Para `glove` y `word2vec`, el evaluador espera `--embedding_model_path` apuntando a un archivo de embeddings en texto plano. Para `sentence_embeddings`, usa `--sentence_embedding_model` y por default carga `sentence-transformers/all-MiniLM-L6-v2`.

Ejemplo de uso sobre un adapter:

```bash
uv run --group vertex python -m src.eval.run_sft_eval \
  --run_dir runs/sft/2026-06-28_1213_ce_uniform \
  --eval_file datasets/final-10k/test.jsonl \
  --max_examples 250 \
  --num_samples 4 \
  --max_new_tokens 8 \
  --temperature 0.6 \
  --top_p 0.9 \
  --top_k 40 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --max_concept_words 3 \
  --embedding_backend sentence_embeddings \
  --sentence_embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding_device cpu
```

`--max_examples` te deja limitar cuántas recetas del dataset se evalúan.

El script escribe:

* `predictions.jsonl`: outputs por ejemplo, muestras y componentes de creatividad.
* `summary.json`: por default solo métricas de creatividad (`plausibility_distance`, `plausibility_score`, `novelty`, `diversity_distance`, `diversity_score`, `ccs`) y las recetas con `local_creativity` mínima y máxima, junto con sus `sampled_outputs`.

### Build y ejecución en Vertex AI

Build de la imagen:

```bash
gcloud builds submit . \
  --tag us-central1-docker.pkg.dev/nlp2026-498021/llm-craft-registry/llm-craft-sft:eval-20260705-short-concepts-v2
```

Job de evaluación en Vertex AI sobre una L4:

```bash
uv run --group vertex python -m src.sft.vertex_submit \
  --run-name qwen3-4b-eval-input-distance-l4-100-tight-v2 \
  --region us-central1 \
  --bucket llm-craft-bucket \
  --image-uri us-central1-docker.pkg.dev/nlp2026-498021/llm-craft-registry/llm-craft-sft:eval-20260705-short-concepts-v2 \
  --module src.eval.run_sft_eval \
  --machine-type g2-standard-8 \
  --accelerator-type NVIDIA_L4 \
  --accelerator-count 1 \
  -- \
  --run_dir gs://llm-craft-bucket/runs/2026-07-01_1950_qwen3_4b_thinking_10k \
  --eval_file gs://llm-craft-bucket/datasets/final-10k/test.jsonl \
  --output_dir gs://llm-craft-bucket/eval_outputs/qwen3_4b_l4_eval_100_input_distance_tight_v2 \
  --max_examples 100 \
  --num_samples 4 \
  --max_new_tokens 8 \
  --temperature 0.6 \
  --top_p 0.9 \
  --top_k 40 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 3 \
  --max_concept_words 3 \
  --alpha 0.8 \
  --lambda_penalty 2.0 \
  --embedding_backend sentence_embeddings \
  --sentence_embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding_device cpu \
  --eval_bf16 true \
  --eval_fp16 false \
  --eval_bnb_4bit_compute_dtype bfloat16 \
  --device cuda
```

### Smoke test local

Con el `default.yaml` actual alcanza con:

```bash
uv run python -m src.sft.train
```

Ese comando usa pocos ejemplos, `max_steps: 2`, `batch_size: 1`, `gradient_accumulation_steps: 1`, y guarda una corrida en `runs/sft/`.

Para usar la variante ponderada tipo concept-set:

```bash
uv run python -m src.sft.train \
  --loss_type concept_set \
  --run_name smoke_tiny_concept_set
```

Para cross entropy agrupada por receta:

```bash
uv run python -m src.sft.train \
  --candidate_weighting uniform \
  --candidate_aggregation expected_logprob \
  --run_name smoke_tiny_ce
```

Los cuatro modos experimentales quedan:

```text
candidate_weighting: uniform   + candidate_aggregation: expected_logprob   # CE agrupada por receta
candidate_weighting: dataset   + candidate_aggregation: expected_logprob   # Soft CE ponderada
candidate_weighting: uniform   + candidate_aggregation: logsumexp_prob     # Concept-set uniforme
candidate_weighting: dataset   + candidate_aggregation: logsumexp_prob     # Concept-set ponderada
```

`loss_type` sigue disponible como alias opcional:

```text
ce                  -> uniform + expected_logprob
soft_ce             -> dataset + expected_logprob
concept_set         -> dataset + logsumexp_prob
concept_set_uniform -> uniform + logsumexp_prob
```

Si definís explícitamente `candidate_weighting` y `candidate_aggregation`, esos valores gobiernan la loss efectiva. `loss_type` queda como alias de compatibilidad para completar ambos ejes cuando no se los fija de forma directa.

No se permiten overrides parciales: si definís `candidate_weighting` o `candidate_aggregation`, tenés que definir ambos. Si no definís ninguno, ambos se derivan automáticamente desde `loss_type`.

Si no definís ninguno de los tres campos, se usan los defaults de la config y el entrenamiento queda en la variante `concept_set`, es decir:

```text
loss_type = concept_set
candidate_weighting = dataset
candidate_aggregation = logsumexp_prob
```

### Comandos para cada experimento

Usando la configuración base de smoke test, estos comandos permiten correr cada variante de la matriz experimental:

#### 1. CE agrupada por receta

```bash
uv run python -m src.sft.train \
  --config configs/sft/default.yaml \
  --run_name ce_uniform \
  --candidate_weighting uniform \
  --candidate_aggregation expected_logprob
```

#### 2. Soft CE ponderada

```bash
uv run python -m src.sft.train \
  --config configs/sft/default.yaml \
  --run_name soft_ce_weighted \
  --candidate_weighting dataset \
  --candidate_aggregation expected_logprob
```

#### 3. Concept-set uniforme

```bash
uv run python -m src.sft.train \
  --config configs/sft/default.yaml \
  --run_name concept_set_uniform \
  --candidate_weighting uniform \
  --candidate_aggregation logsumexp_prob
```

#### 4. Concept-set ponderada

```bash
uv run python -m src.sft.train \
  --config configs/sft/default.yaml \
  --run_name concept_set_weighted \
  --candidate_weighting dataset \
  --candidate_aggregation logsumexp_prob
```

#### 5. Concept-set ponderada + rationales

Entrena el mismo objetivo de concepto y agrega un término auxiliar sobre el campo `rationale`
de cada candidato. Los candidatos sin rationale siguen entrenando solo el concepto. Esta
variante genera primero el concepto y luego la explicación:

```text
Input A: fire
Input B: water
Final concept: steam
Rationale: Fire heats water until it becomes steam.
```

```bash
uv run python -m src.sft.train \
  --config configs/sft/qwen05b_10k_rationale_example.yaml
```

Para entrenar la variante en la que el rationale aparece antes y el concepto final queda
condicionado por la explicación:

```text
Input A: fire
Input B: water
Rationale: Fire heats water until it becomes steam.
Final concept: steam
```

usar:

```bash
uv run python -m src.sft.train \
  --config configs/sft/qwen05b_10k_rationale_first_example.yaml
```

El término auxiliar queda controlado por:

```text
rationale_loss_weight = 0.2
length_normalize_rationale_logprob = true
rationale_position = output_before_rationale | output_after_rationale
```

Los mismos cuatro experimentos también pueden invocarse con aliases legacy de `loss_type`:

```bash
uv run python -m src.sft.train --config configs/sft/default.yaml --run_name ce_uniform --loss_type ce
uv run python -m src.sft.train --config configs/sft/default.yaml --run_name soft_ce_weighted --loss_type soft_ce
uv run python -m src.sft.train --config configs/sft/default.yaml --run_name concept_set_uniform --loss_type concept_set_uniform
uv run python -m src.sft.train --config configs/sft/default.yaml --run_name concept_set_weighted --loss_type concept_set
```

### Entrenamiento con QLoRA

Para una corrida real con Qwen 4B hace falta una máquina con CUDA. En CPU la carga y el backward de un modelo 4B son extremadamente lentos.

```bash
uv run python -m src.sft.train \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --train_path datasets/final-small-dataset/train.jsonl \
  --dev_path datasets/final-small-dataset/dev.jsonl \
  --output_dir runs/sft \
  --run_name qwen4b_concept_set \
  --loss_type concept_set \
  --load_in_4bit true \
  --bf16 true \
  --fp16 false \
  --gradient_checkpointing true \
  --max_train_examples 32 \
  --max_dev_examples 16 \
  --max_steps 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --eval_steps 5 \
  --save_steps 5 \
  --logging_steps 1
```

Para probar Qwen3 4B con `thinking mode` y el chat template del tokenizer:

```bash
uv run python -m src.sft.train \
  --config configs/sft/qwen3_4b_thinking_10k_example.yaml \
  --max_train_examples 32 \
  --max_dev_examples 16 \
  --max_steps 10 \
  --run_name qwen3_4b_thinking_smoke
```

La config `qwen3_4b_thinking_10k_example.yaml` usa `Qwen/Qwen3-4B-Thinking-2507`, activa `prompt_format: qwen_chat` y mantiene la supervisión solo sobre el concepto final del mensaje `assistant`.

Con `prompt_format: qwen_chat`, el contenido lógico del prompt queda así:

```text
system: You combine two concepts into one resulting concept.
user:
Given two concepts, combine them into one resulting concept.

Concept A: fire
Concept B: water

Return only the resulting concept.
assistant: steam
```

Durante el entrenamiento, `train.py` ahora imprime al inicio el plan (`train_recipes`, `dev_recipes`, `total_steps`) y en cada log de train agrega `elapsed` y `eta`, tanto localmente como en los logs de Vertex.

### Reanudar desde checkpoint

```bash
uv run python -m src.sft.train \
  --config configs/sft/default.yaml \
  --resume_from_checkpoint runs/sft/<run_id>/checkpoints/checkpoint-000002 \
  --max_steps 4 \
  --run_name smoke_resume
```

Los checkpoints guardan adapter, tokenizer y estado de `accelerate` para recuperar optimizer, scheduler y RNG.

### Losses implementadas

Todas las variantes promedian por receta y difieren solo en el esquema de pesos y la agregación. Para cada receta \(x_n\) con candidatos \(c_{n,i}\) y log-probabilidades \(\ell_{n,i} = \log p_\theta(c_{n,i} \mid x_n)\):

```text
expected_logprob: L_n = -Sum_i alpha_{n,i} * ell_{n,i}
logsumexp_prob:   L_n = -log Sum_i alpha_{n,i} * exp(ell_{n,i})
```

La log-probabilidad se calcula solo sobre los tokens del concepto final, no sobre el prompt. Por default se usa la probabilidad autoregresiva completa del concepto, es decir, la suma de log-probabilidades de sus tokens. Si activás `length_normalize_concept_logprob: true`, pasás a una variante experimental que promedia por longitud del concepto antes de agregar por receta.

El collator siempre preserva todos los candidatos de una receta dentro del mismo batch; `per_device_*_batch_size` cuenta recetas, no candidatos tokenizados.

### Dataset esperado

El formato principal es JSONL con `candidate_outputs`:

```json
{"input_a":"fire","input_b":"water","candidate_outputs":[{"output":"steam","source":"observed","rank":1},{"output":"vapor","source":"teacher","rank":2}]}
```

Si un candidato trae el campo configurado con `--weight_field weight`, se usa como peso. Si falta, `--weight_fallback inverse_rank` asigna pesos proporcionales a `1/rank` y los normaliza dentro de cada receta; `--weight_fallback uniform` reparte masa uniforme. Con `merge_duplicate_recipes: true`, filas JSONL repetidas con el mismo `(input_a, input_b)` se fusionan en una sola receta, se deduplican candidatos por `output` sumando su masa y luego se renormaliza dentro de la receta.

`ce_target` se conserva solo por compatibilidad hacia atrás y hoy no altera la selección de candidatos: incluso en la CE agrupada entran todos los candidatos aceptables de la receta.

También se aceptan filas legacy con `outputs` u `output`; `dataset.py` las adapta internamente a candidatos.

### Outputs de una corrida

Cada corrida se guarda en `runs/sft/<timestamp>_<run_name_o_model_loss>/`:

```text
config.yaml
command.txt
git_info.json
data_fingerprint.json
metrics.jsonl
train_losses.jsonl
eval_losses.jsonl
plots/
  train_loss.png
  dev_loss.png
  losses_combined.png
checkpoints/
  checkpoint-000002/
best_adapter/
final_adapter/
tokenizer/
trainer_state.json
```

Para revisar rápido una corrida:

```bash
cat runs/sft/<run_id>/metrics.jsonl
cat runs/sft/<run_id>/trainer_state.json
ls runs/sft/<run_id>/checkpoints
```

### Tests de SFT

Los tests unitarios nuevos cubren carga JSONL, normalización de pesos, merge por receta, máscara por offsets, candidatos variables y las cuatro variantes de loss, además del smoke train opcional:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/sft -q
```

El smoke que descarga un modelo está desactivado por defecto. Para correrlo explícitamente:

```bash
RUN_SFT_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/sft/test_smoke_train.py -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` evita que pytest cargue plugins globales del sistema que no pertenecen al proyecto.

---

## SFT en Vertex AI (Google Cloud)

El mismo `src/sft/train.py` corre como **Custom Training Job** en Vertex AI sin modificaciones: Vertex monta el bucket de GCS en `/gcs/<bucket>/`, así que los datos entran desde `datasets/` y la corrida (adapters, checkpoints, métricas y plots) sale a `runs/`.

```bash
# 1. Subir datos al bucket (si no están ya)
gsutil -m cp datasets/final-10k/train.jsonl datasets/final-10k/dev.jsonl \
  gs://llm-craft-bucket/datasets/final-10k/

# 2. Buildear + pushear la imagen a Artifact Registry (Cloud Build)
gcloud builds submit --config cloudbuild.yaml --project nlp2026-498021

# 3. Lanzar el entrenamiento (1x NVIDIA L4)
uv run --group vertex python -m src.sft.vertex_submit \
  --run-name qwen05b-10k --config configs/sft/qwen05b_10k_example.yaml

# Variante Qwen3 4B Thinking con chat template de Qwen
uv run --group vertex python -m src.sft.vertex_submit \
  --run-name qwen3-4b-thinking-10k \
  --config configs/sft/qwen3_4b_thinking_10k_example.yaml
```

También podés usar el helper del repo:

```bash
./scripts/run_vertex_qwen3_thinking.sh
```

Ese script prioriza ADC si ya corriste `gcloud auth application-default login`; si no encuentra ADC, usa automáticamente `nlp2026-498021-8c813796c042.json` si está en la raíz del repo.
Si interrumpís el proceso con `Ctrl+C`, el submitter intenta cancelar el `CustomJob` remoto automáticamente.

`train.py` es config-driven: la imagen lleva los YAML de `configs/sft/` horneados y `vertex_submit.py` sobrescribe las rutas para que apunten al mount `/gcs/`. La dependencia `google-cloud-aiplatform` vive en el grupo opt-in `vertex`, por lo que el entrenamiento local y la imagen no la instalan.

La guía completa (prerrequisitos, permisos, cuota de GPU, overrides y descarga de adapters) está en [vertex_training.md](docs/codigo/vertex_training.md).

---

## Frontend

La interfaz jugable vive en `apps/web` como una app Next.js. El combinador consulta recetas en Postgres y usa Vertex AI con `VERTEX_API_KEY` o `GOOGLE_APPLICATION_CREDENTIALS` para generar y guardar combinaciones nuevas cuando el par no existe en la base. `Sandbox` y `Goal` permiten elegir el modelo Gemini del combinador (`Gemini 2.5 Flash`, `Gemini 2.5 Pro` o `Gemini 2.5 Flash Lite`); esa seleccion solo aplica a combinaciones generadas por modelo, porque las recetas conocidas de `final-10k` siempre tienen prioridad para mantener resultados deterministas. Incluye registro/login mock con credenciales seeded `admin/admin`, menu de modos (`Sandbox`, `Goal` y `Agent Test`), metas alcanzables por profundidad con inventario inicial variable, controles internos para reiniciar o generar otra meta en `Goal`, un modo de prueba donde un agente usa el combinador hasta 20 mezclas, `DPO test mode`, perfil con logros destacados y leaderboard para objetivos completados.

```bash
cd apps/web
npm ci
npm run dev
```

Para configurar Vertex en desarrollo:

```bash
cp .env.example .env.local
```

Para validar el frontend:

```bash
npm run typecheck
npm run test
npm run build
```

Más detalles: [frontend_next_app.md](docs/codigo/frontend_next_app.md).

---

## Documentación del Proyecto

Para más detalles teóricos y de diseño, consulte:
* [adr_data_pipeline.md](docs/codigo/adr_data_pipeline.md): Architecture Decision Record (ADR) con las decisiones del pipeline.
* [data_pipeline.md](docs/codigo/data_pipeline.md): Especificaciones técnicas de la limpieza, hashes y formato SFT.
* [data_normalization.md](docs/codigo/data_normalization.md): Proceso de extracción inicial de datasets crudos.
* [sft_qlora_pipeline.md](docs/codigo/sft_qlora_pipeline.md): Diseño e implementación de la pipeline SFT con LoRA/QLoRA, losses y outputs.
* [vertex_training.md](docs/codigo/vertex_training.md): Entrenar `src/sft/train.py` en Vertex AI con imagen propia en Artifact Registry y datos en GCS.
* [frontend_next_app.md](docs/codigo/frontend_next_app.md): Guía para ejecutar, validar y extender la app Next.js jugable.
* [destilacion_creatividad_composicional.md](docs/informe/destilacion_creatividad_composicional.md): Paper de diseño del proyecto de investigación.
