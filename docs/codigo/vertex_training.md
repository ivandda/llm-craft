# SFT en Vertex AI (Google Cloud)

Esta guia cubre como correr `src/sft/train.py` como **Custom Training Job** en
Vertex AI, usando una imagen propia en Artifact Registry y leyendo/escribiendo
datos directamente del bucket de GCS. El script de entrenamiento no se modifica:
Vertex monta el bucket via Cloud Storage FUSE en `/gcs/<bucket>/`, asi que los
paths locales de `train.py` apuntan al bucket sin codigo extra.

## Recursos del proyecto

| Recurso | Valor |
|---|---|
| Project ID | `nlp2026-498021` |
| Region | `us-central1` |
| Bucket | `gs://llm-craft-bucket` (carpetas `datasets/` y `runs/`) |
| Artifact Registry | `llm-craft-registry` |
| Imagen | `us-central1-docker.pkg.dev/nlp2026-498021/llm-craft-registry/llm-craft-sft:latest` |
| Maquina | `g2-standard-8` + `1x NVIDIA_L4` |

## Archivos involucrados

- `Dockerfile`: imagen CUDA 12.4 con las deps del lockfile (`uv sync --frozen --no-dev`).
- `.dockerignore`: mantiene el build liviano (excluye `datasets/`, `runs/`, `.venv/`, `apps/`, etc.).
- `cloudbuild.yaml`: buildea y pushea la imagen a Artifact Registry con Cloud Build.
- `src/sft/vertex_submit.py`: crea el `CustomJob` con el SDK `google-cloud-aiplatform`.

La dependencia `google-cloud-aiplatform` vive en el grupo opt-in `vertex` de
`pyproject.toml`, asi que **no** se instala en el entrenamiento local ni en la
imagen; solo cuando se la pide explicitamente.

## Prerrequisitos (una sola vez)

```bash
# 1. APIs
gcloud services enable \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project nlp2026-498021

# 2. Credenciales locales (las usa el SDK en vertex_submit.py via ADC)
gcloud auth application-default login
```

### Permiso de escritura en el bucket (por si el primer job falla)

El contenedor de entrenamiento corre como el **Custom Code Service Agent** de
Vertex (`service-PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com`).
Ese service agent **no existe hasta que se lanza el primer Custom Job**, asi que
el `iam ch` falla con `does not exist` si se corre antes. El orden correcto es:

1. Lanzar el primer job (Vertex crea el service agent al ejecutarlo).
2. Si el job entrena pero **falla al guardar** el adapter en GCS, darle el rol
   (ya existe el agent) y reintentar:

```bash
# Reemplazar PROJECT_NUMBER con el resultado de: gcloud projects describe nlp2026-498021 --format='value(projectNumber)'
gsutil iam ch \
  serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://llm-craft-bucket
```

> En muchos proyectos el agent ya tiene acceso a los buckets del mismo proyecto y
> este paso no hace falta. Solo aplicalo si el job falla al escribir en `runs/`.

## Flujo de entrenamiento

### 1. Subir los datos al bucket

```bash
gsutil -m cp datasets/final-10k/train.jsonl datasets/final-10k/dev.jsonl \
  gs://llm-craft-bucket/datasets/final-10k/
```

### 2. Buildear y pushear la imagen (Cloud Build)

```bash
gcloud builds submit --config cloudbuild.yaml --project nlp2026-498021
```

Para imagenes inmutables por commit:

```bash
gcloud builds submit --config cloudbuild.yaml --project nlp2026-498021 \
  --substitutions _TAG=$(git rev-parse --short HEAD)
```

### 3. Lanzar el entrenamiento

`train.py` es config-driven: la imagen lleva `configs/sft/default.yaml` horneado y
`vertex_submit.py` lo pasa con `--config`, sobrescribiendo las rutas para que
apunten al mount `/gcs/`.

```bash
uv run --group vertex python -m src.sft.vertex_submit --run-name <run-name>
```

Defaults del submit (todos overrideables por flag):

- `--config configs/sft/default.yaml` (define modelo, loss e hiperparametros)
- `--train-path /gcs/llm-craft-bucket/datasets/final-10k/train.jsonl`
- `--dev-path   /gcs/llm-craft-bucket/datasets/final-10k/dev.jsonl`
- `--output_dir /gcs/llm-craft-bucket/runs` (train.py crea adentro un
  subdir con timestamp + modelo + loss + run_name)
- Maquina: `g2-standard-8` + `1x NVIDIA_L4` (`--machine-type`, `--accelerator-count`)
- `--model-name` sobrescribe `model_name_or_path` del YAML; `--max-steps` cap de pasos.

Cualquier campo de `train.py` (subrayado, p. ej. `--lora_r`, `--learning_rate`,
`--load_in_4bit`) se pasa verbatim despues de `--`:

```bash
uv run --group vertex python -m src.sft.vertex_submit --run-name <run-name> \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  -- --num_train_epochs 3 --load_in_4bit false --lora_r 16
```

> Para validar el pipeline sin gastar GPU, correr un smoke en CPU:
> `--machine-type n1-standard-8 --accelerator-count 0`.

### 4. Descargar el adapter entrenado

El run queda en un subdir con timestamp; listar y bajar el que corresponda:

```bash
gsutil ls gs://llm-craft-bucket/runs/
gsutil -m cp -r gs://llm-craft-bucket/runs/<run-id> .
```

Dentro de cada run: `best_adapter/`, `final_adapter/`, `checkpoints/`,
`config.yaml`, `metrics.jsonl`, `eval_losses.jsonl` y `plots/`.

## Entrenamiento local vs Vertex

El mismo `src/sft/train.py` corre en ambos lados. Local no necesita el grupo
`vertex`:

```bash
uv run python -m src.sft.train \
  --config configs/sft/default.yaml \
  --train_path datasets/final-10k/train.jsonl \
  --dev_path datasets/final-10k/dev.jsonl \
  --output_dir runs/local-smoke \
  --run_name local-smoke \
  --max_steps 2
```
