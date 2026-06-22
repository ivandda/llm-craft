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
* **Configuración**: [pipeline_config.yaml](configs/pipeline_config.yaml) (permite controlar proporciones de splits, exclusión de copias de identidad, prompt templates y tamaños de sets de evaluación).

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
* **Qué hace**: Convierte todos los conceptos a minúsculas, agrupa las recetas duplicadas por par de entrada, identifica y calcula conflictos (`is_conflicting_pair`), y asigna cada combinación a `train`/`dev`/`test` usando hashes del par de entrada para evitar filtración de datos (leakage). Genera `recipe_canonical_v0.jsonl` y `clean_metrics.json`.

### 3. Exportación para Fine-Tuning (SFT) (Capa de Oro)
Prepara los datasets conversacionales para el entrenamiento supervisado.
```bash
uv run python -m src.data.export_sft
```
* **Qué hace**: Exporta dos variantes conversacionales en formato `messages`: `sft_clean` (recetas sin conflictos ni copias triviales) y `sft_all` (todas las recetas válidas incluyendo conflictos), aplicando el prompt template configurado y los metadatos enriquecidos (`pair_id`, `recipe_id`, etc.).

### 4. Exportación de Conjuntos de Evaluación
Genera los datasets estructurados con respuestas alternativas válidas conocidas para evaluar al estudiante.
```bash
uv run python -m src.data.export_eval
```
* **Qué hace**: Utiliza un algoritmo de *Reservoir Sampling* de un solo paso de lectura para extraer muestras aleatorias deterministas para evaluación (`eval_dev_1k`, `eval_test_1k`, etc.) con un uso de memoria de menos de 10 MB. Cada registro lista todas las respuestas válidas conocidas (`known_outputs`) asociadas al par para permitir un score preciso.

---

## SFT y Google Colab

### Preparar muestras para entrenamiento

```bash
uv run python -m src.sft.prepare_train_eval_data \
  --train-input datasets/processed/sft_clean_train.jsonl \
  --eval-input datasets/processed/sft_clean_dev.jsonl \
  --train-output artifacts/data/sft_clean_train_sample_8000.jsonl \
  --eval-output artifacts/data/sft_clean_dev_sample_2000.jsonl \
  --train-sample-size 8000 \
  --eval-sample-size 2000 \
  --seed 42
```

### Crear zip para Colab

```bash
uv run python -m src.sft.create_colab_zip \
  --train-file artifacts/data/sft_clean_train_sample_8000.jsonl \
  --eval-file artifacts/data/sft_clean_dev_sample_2000.jsonl \
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
  --train-file artifacts/data/sft_clean_train_sample_8000.jsonl \
  --eval-file artifacts/data/sft_clean_dev_sample_2000.jsonl \
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

La guía completa está en [sft_training_colab.md](docs/codigo/sft_training_colab.md).

---

## Documentación del Proyecto

Para más detalles teóricos y de diseño, consulte:
* [adr_data_pipeline.md](docs/codigo/adr_data_pipeline.md): Architecture Decision Record (ADR) con las decisiones del pipeline.
* [data_pipeline.md](docs/codigo/data_pipeline.md): Especificaciones técnicas de la limpieza, hashes y formato SFT.
* [data_normalization.md](docs/codigo/data_normalization.md): Proceso de extracción inicial de datasets crudos.
* [sft_training_colab.md](docs/codigo/sft_training_colab.md): Comandos para preparar muestras SFT, empaquetar Colab, entrenar y predecir.
* [destilacion_creatividad_composicional.md](docs/informe/destilacion_creatividad_composicional.md): Paper de diseño del proyecto de investigación.