# SFT: preparacion, entrenamiento y Colab

Esta guia cubre los scripts de SFT del proyecto y el flujo para crear un zip
listo para Google Colab.

## 1. Preparar muestras de train y eval

El script `src/sft/prepare_train_eval_data.py` crea muestras deterministicas a
partir de los datasets conversacionales completos.

Entrada por defecto:

```text
datasets/processed/recipes_train.jsonl
datasets/processed/recipes_dev.jsonl
```

Salida por defecto:

```text
artifacts/data/recipes_train_sample_8000.jsonl
artifacts/data/recipes_dev_sample_2000.jsonl
```

Comando recomendado:

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

El muestreo usa `metadata.recipe_id` y un hash estable con `--seed`, por lo que
el mismo comando produce siempre las mismas filas.

## 2. Crear el zip para Google Colab

El script `src/sft/create_colab_zip.py` arma un zip chico con:

- `src/`
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `docs/`
- los JSONL de train y eval indicados
- `datasets/processed/eval_dev_all.jsonl` para evaluacion batch durante desarrollo

No incluye `.venv`, caches, checkpoints ni modelos entrenados.

Antes de crear el zip, asegurate de que exista
`datasets/processed/eval_dev_all.jsonl`. Si falta, generalo con:

```bash
uv run python -m src.data.run_pipeline
uv run python -m src.data.export_eval
```

Comando recomendado:

```bash
uv run python -m src.sft.create_colab_zip \
  --train-file artifacts/data/recipes_train_sample_8000.jsonl \
  --eval-file artifacts/data/recipes_dev_sample_2000.jsonl \
  --output-path artifacts/colab/llm-craft-sft-colab.zip
```

El zip queda en:

```text
artifacts/colab/llm-craft-sft-colab.zip
```

## 3. Subir y descomprimir en Colab

En Colab, subi `llm-craft-sft-colab.zip` al panel de archivos o montalo desde
Google Drive. Si lo subiste al directorio actual de Colab, ejecuta:

```bash
!unzip -q llm-craft-sft-colab.zip -d /content
%cd /content/llm-craft-colab
```

Si el zip esta en Drive, por ejemplo en `MyDrive/llm-craft-sft-colab.zip`:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!unzip -q /content/drive/MyDrive/llm-craft-sft-colab.zip -d /content
%cd /content/llm-craft-colab
```

## 4. Instalar dependencias en Colab

Instala `uv` y sincroniza el entorno:

```bash
!pip install -q uv
!uv sync
```

Si Colab tiene problemas con el lock por plataforma CUDA, usa instalacion
resuelta en Colab:

```bash
!uv sync --upgrade
```

## 5. Entrenar

Entrenamiento LoRA:

```bash
!uv run python -m src.sft.train \
  --train-file artifacts/data/recipes_train_sample_8000.jsonl \
  --eval-file artifacts/data/recipes_dev_sample_2000.jsonl \
  --output-dir artifacts/sft/smollm2-clean-lora \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --lora-mode lora \
  --max-steps 100
```

Entrenamiento QLoRA en GPU CUDA:

```bash
!uv run python -m src.sft.train \
  --train-file artifacts/data/recipes_train_sample_8000.jsonl \
  --eval-file artifacts/data/recipes_dev_sample_2000.jsonl \
  --output-dir artifacts/sft/smollm2-clean-qlora \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --lora-mode qlora \
  --max-steps 100
```

El script guarda el adapter, tokenizer, checkpoints y `run_config.json` en
`--output-dir`. Como se usa LoRA/QLoRA, no se guarda una copia completa del
modelo base.

## 6. Probar inferencia

Despues de entrenar:

```bash
!uv run python -m src.sft.predict \
  --adapter-dir artifacts/sft/smollm2-clean-lora \
  --input-a fire \
  --input-b water
```

Tambien se puede pasar un prompt manual:

```bash
!uv run python -m src.sft.predict \
  --adapter-dir artifacts/sft/smollm2-clean-lora \
  --prompt "Given two concepts, combine them into one resulting concept.

Concept A: fire
Concept B: water

Return only the resulting concept."
```

## 7. Evaluar en batch

Evaluar el adapter entrenado contra el set dev estructurado con respuestas conocidas:
si `datasets/processed/eval_dev_all.jsonl` no existe, correr primero
`uv run python -m src.data.export_eval`.

```bash
!uv run python -m src.eval.run_sft_eval \
  --eval-file datasets/processed/eval_dev_all.jsonl \
  --output-file artifacts/eval/smollm2-clean-lora-dev.jsonl \
  --adapter-dir artifacts/sft/smollm2-clean-lora
```

Para una prueba rapida antes de correr el set completo:

```bash
!uv run python -m src.eval.run_sft_eval \
  --eval-file datasets/processed/eval_dev_all.jsonl \
  --output-file artifacts/eval/smollm2-clean-lora-smoke.jsonl \
  --adapter-dir artifacts/sft/smollm2-clean-lora \
  --limit 5
```

El comando escribe un JSONL con una prediccion por par e imprime:
`canonical_accuracy`, `known_output_accuracy` y `empty_predictions`.

Reservar `eval_test_all.jsonl` para la corrida final del experimento:

```bash
!uv run python -m src.eval.run_sft_eval \
  --eval-file datasets/processed/eval_test_all.jsonl \
  --output-file artifacts/eval/smollm2-clean-lora-final-test.jsonl \
  --adapter-dir artifacts/sft/smollm2-clean-lora
```

## Resumen de scripts

- `prepare_train_eval_data.py`: crea muestras deterministicas de train y eval.
- `create_colab_zip.py`: empaqueta codigo, docs y datos sampleados para Colab.
- `train.py`: entrena un adapter LoRA o QLoRA sobre un modelo chat base.
- `predict.py`: carga el modelo base y el adapter entrenado para generar una
  respuesta.
- `run_sft_eval.py`: evalua un modelo base o adapter contra `known_outputs`.
