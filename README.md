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
* **Qué hace**: Utiliza *Reservoir Sampling* de un solo paso de lectura para extraer muestras aleatorias deterministas (`eval_dev_1k`, `eval_test_1k`, etc.) con bajo uso de memoria. Cada registro es mínimo y lista las respuestas válidas conocidas (`known_outputs`) asociadas al par.
* **Tamaño completo**: En `evaluation_export.sizes`, usar `all` en lugar de un número para exportar todo el split (`eval_dev_all.jsonl`, `eval_test_all.jsonl`).

---

## SFT y Google Colab

### Preparar muestras para entrenamiento

```bash
uv run python -m src.sft.prepare_train_eval_data \
  --train-input datasets/processed/recipes_train.jsonl \
  --eval-input datasets/processed/recipes_dev.jsonl \
  --train-output artifacts/data/recipes_train_sample_8000.jsonl \
  --eval-output artifacts/data/recipes_dev_sample_2000.jsonl \
  --train-sample-size 8000 \
  --eval-sample-size 2000 \
  --seed 42
```

### Crear zip para Colab

El zip incluye `datasets/processed/eval_dev_1k.jsonl` para evaluacion batch.
Si todavia no existe, generarlo con:

```bash
uv run python -m src.data.run_pipeline
uv run python -m src.data.export_eval
```

```bash
uv run python -m src.sft.create_colab_zip \
  --train-file artifacts/data/recipes_train_sample_8000.jsonl \
  --eval-file artifacts/data/recipes_dev_sample_2000.jsonl \
  --output-path artifacts/colab/llm-craft-sft-colab.zip
```

En Colab, subir el zip y descomprimirlo:

```bash
!unzip -q llm-craft-sft-colab.zip -d /content
%cd /content/llm-craft-colab
!pip install -q uv
!uv sync
```

### Entrenar

```bash
uv run python -m src.sft.train \
  --train-file artifacts/data/recipes_train_sample_8000.jsonl \
  --eval-file artifacts/data/recipes_dev_sample_2000.jsonl \
  --output-dir artifacts/sft/smollm2-clean-lora \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --lora-mode lora \
  --max-steps 100
```

### Inferencia

```bash
uv run python -m src.sft.predict \
  --adapter-dir artifacts/sft/smollm2-clean-lora \
  --input-a fire \
  --input-b water
```

### Evaluación batch

Durante el desarrollo, usar `eval_dev_1k.jsonl` para comparar variantes sin tocar el test final.
Si falta ese archivo, correr primero `uv run python -m src.data.export_eval`.

Evaluar el modelo base contra respuestas conocidas de dev:

```bash
uv run python -m src.eval.run_sft_eval \
  --eval-file datasets/processed/eval_dev_1k.jsonl \
  --output-file artifacts/eval/smollm2-base-dev.jsonl \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct
```

Evaluar un adapter LoRA entrenado en dev:

```bash
uv run python -m src.eval.run_sft_eval \
  --eval-file datasets/processed/eval_dev_1k.jsonl \
  --output-file artifacts/eval/smollm2-clean-lora-dev.jsonl \
  --adapter-dir artifacts/sft/smollm2-clean-lora
```

El comando genera un JSONL con predicciones por ejemplo e imprime métricas agregadas:
`canonical_accuracy`, `known_output_accuracy` y `empty_predictions`.

Reservar `eval_test_1k.jsonl` para la evaluación final.

La guía completa está en [sft_training_colab.md](docs/codigo/sft_training_colab.md).

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
* [sft_training_colab.md](docs/codigo/sft_training_colab.md): Comandos para preparar muestras SFT, empaquetar Colab, entrenar y predecir.
* [frontend_next_app.md](docs/codigo/frontend_next_app.md): Guía para ejecutar, validar y extender la app Next.js jugable.
* [destilacion_creatividad_composicional.md](docs/informe/destilacion_creatividad_composicional.md): Paper de diseño del proyecto de investigación.
