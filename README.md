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

Smoke test real mínimo:
```bash
uv run python -m src.data.enrich_multi_output --splits train --limit 3
```

La generación completa queda como paso manual para controlar costo:
```bash
uv run python -m src.data.enrich_multi_output
```

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

La pipeline de SFT vive aislada en `src/sft/` y entrena modelos causales autoregresivos sobre datasets JSONL con candidatos múltiples por receta. Soporta LoRA/QLoRA, `concept_set` loss y cross entropy clásica (`ce`) seleccionando un candidato por receta.

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
  losses.py      # concept_set loss y CE sobre tokens del concepto final.
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

### Smoke test local

Con el `default.yaml` actual alcanza con:

```bash
uv run python -m src.sft.train
```

Ese comando usa pocos ejemplos, `max_steps: 2`, `batch_size: 1`, `gradient_accumulation_steps: 1`, y guarda una corrida en `runs/sft/`.

Para forzar `concept_set`:

```bash
uv run python -m src.sft.train \
  --loss_type concept_set \
  --run_name smoke_tiny_concept_set
```

Para cross entropy clásica:

```bash
uv run python -m src.sft.train \
  --loss_type ce \
  --ce_target rank1 \
  --run_name smoke_tiny_ce
```

`ce_target` puede ser:

```text
rank1     # candidato con menor rank; default recomendado para CE
observed  # primer candidato con source == observed; si no hay, cae al primero
first     # primer candidato de la lista JSONL
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

`concept_set` usa todos los candidatos de una receta. Para cada receta \(x_n\) con candidatos \(c_{n,i}\) y pesos normalizados \(w_{n,i}\):

```text
L_n = -logsumexp_i(log w_{n,i} + log p_theta(c_{n,i} | x_n))
```

La log-probabilidad se calcula solo sobre los tokens del concepto final, no sobre el prompt.

`ce` selecciona un único candidato por receta y optimiza:

```text
L_n = -log p_theta(c_n | x_n)
```

### Dataset esperado

El formato principal es JSONL con `candidate_outputs`:

```json
{"input_a":"fire","input_b":"water","candidate_outputs":[{"output":"steam","source":"observed","rank":1},{"output":"vapor","source":"teacher","rank":2}]}
```

Si un candidato trae el campo configurado con `--weight_field weight`, se usa como peso. Si falta, `--weight_fallback inverse_rank` asigna pesos proporcionales a `1/rank` y los normaliza dentro de cada receta; `--weight_fallback uniform` reparte masa uniforme.

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

Los tests unitarios nuevos cubren carga JSONL, normalización de pesos, máscara por offsets, candidatos variables, `concept_set_loss` diferenciable y smoke train opcional:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/sft -q
```

El smoke que descarga un modelo está desactivado por defecto. Para correrlo explícitamente:

```bash
RUN_SFT_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/sft/test_smoke_train.py -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` evita que pytest cargue plugins globales del sistema que no pertenecen al proyecto.

---

## Frontend

La interfaz jugable vive en `apps/web` como una app Next.js preparada para conectar modelos más adelante mediante contratos mock tipados. Incluye registro/login mock en memoria con credenciales seeded `admin/admin`, menu de modos (`Sandbox` y `Goal`), perfil con logros destacados y leaderboard mock para objetivos completados.

```bash
cd apps/web
npm ci
npm run dev
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
* [frontend_next_app.md](docs/codigo/frontend_next_app.md): Guía para ejecutar, validar y extender la app Next.js jugable.
* [destilacion_creatividad_composicional.md](docs/informe/destilacion_creatividad_composicional.md): Paper de diseño del proyecto de investigación.


---

# Batch info:

```bash
(llm-craft) ➜  llm-craft git:(feature/enrich-dataset) ✗ uv run python -m src.data.enrich_teacher \
  --mode batch-submit \
  --batch-input-uri gs://llm-craft-nlp2026-498021/batch_input/batch_requests.jsonl \
  --batch-output-uri gs://llm-craft-nlp2026-498021/batch_output/ \
  --batch-display-name llm-craft-teacher-ranked-v2
{
  "name": "projects/486944883203/locations/us-central1/batchPredictionJobs/612764435819266048",
  "display_name": "llm-craft-teacher-ranked-v2",
  "state": "JOB_STATE_PENDING",
  "create_time": "2026-06-27T20:06:18.146878Z",
  "update_time": "2026-06-27T20:06:18.146878Z",
  "model": "publishers/google/models/gemini-2.5-flash",
  "src": {
    "format": "jsonl",
    "gcs_uri": [
      "gs://llm-craft-nlp2026-498021/batch_input/batch_requests.jsonl"
    ]
  },
  "dest": {
    "format": "jsonl",
    "gcs_uri": "gs://llm-craft-nlp2026-498021/batch_output/"
  },
  "is_terminal_state": false
}
(llm-craft) ➜  llm-craft git:(feature/enrich-dataset) ✗ 
```
