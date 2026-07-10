# Qwen Combiner VM Runbook

Runbook para crear una VM temporal con GPU NVIDIA L4, levantar la API de inferencia de Qwen Combiner y configurar la web local para pegarle por IP efimera. Esta VM no usa IP fija.

## Objetivo

- Proyecto: `nlp2026-498021`
- VM: `qwen-combiner-test`
- Maquina: `g2-standard-4`
- GPU: `1x NVIDIA L4`
- Disco: `150GB pd-ssd`
- Base model: `Qwen/Qwen3-4B-Thinking-2507`
- Adapter: `gs://llm-craft-bucket/runs/2026-07-09_0641_qwen3_4b_dpo_softce/best_adapter`
- API esperada: `http://<IP_EFIMERA>:8000/v1/chat/completions`

## Prerequisitos

En la maquina local donde corra el agente:

1. Instalar Google Cloud CLI.
2. Autenticarse:

```powershell
gcloud.cmd auth login
gcloud.cmd auth application-default login
```

3. Verificar cuenta activa:

```powershell
gcloud.cmd auth list --filter=status:ACTIVE --format="value(account)"
```

4. Configurar proyecto:

```powershell
gcloud.cmd config set project nlp2026-498021
```

## Variables Locales

En PowerShell:

```powershell
$project = "nlp2026-498021"
$vm = "qwen-combiner-test"
$machine = "g2-standard-4"
$tag = "qwen-combiner"
$firewall = "allow-qwen-combiner-8000"
$apiKey = "dev-qwen-key"
$modelName = "qwen3-4b-dpo-softce"
```

## Elegir Zona

Intentar primero `us-central1-c`. Si falla por `ZONE_RESOURCE_POOL_EXHAUSTED`, probar zonas alternativas con L4. En ejecuciones anteriores, `us-west1-a` tuvo stock.

```powershell
$zones = @(
  "us-central1-c",
  "us-central1-a",
  "us-central1-b",
  "us-east1-b",
  "us-east1-c",
  "us-east1-d",
  "us-east4-a",
  "us-east4-c",
  "us-west1-a",
  "us-west1-b",
  "us-west1-c"
)

$createdZone = $null

foreach ($zone in $zones) {
  Write-Host "Trying $zone"
  gcloud.cmd compute instances create $vm `
    --project $project `
    --zone $zone `
    --machine-type $machine `
    --accelerator type=nvidia-l4,count=1 `
    --maintenance-policy TERMINATE `
    --provisioning-model STANDARD `
    --boot-disk-size 150GB `
    --boot-disk-type pd-ssd `
    --image-family ubuntu-2204-lts `
    --image-project ubuntu-os-cloud `
    --scopes cloud-platform `
    --tags $tag

  if ($LASTEXITCODE -eq 0) {
    $createdZone = $zone
    break
  }
}

if (-not $createdZone) {
  throw "Could not create VM in any configured zone."
}

gcloud.cmd config set compute/zone $createdZone
Write-Host "CREATED_ZONE=$createdZone"
```

If the VM already exists, get its zone instead:

```powershell
$createdZone = (
  gcloud.cmd compute instances list `
    --project $project `
    --filter="name=($vm)" `
    --format="value(zone.basename())"
)
```

## Firewall Temporal

Restringir el puerto `8000` a la IP publica actual de la maquina que va a consumir la API:

```powershell
$publicIp = (Invoke-RestMethod -Uri "https://ifconfig.me/ip").Trim()

gcloud.cmd compute firewall-rules create $firewall `
  --project $project `
  --allow tcp:8000 `
  --source-ranges "$publicIp/32" `
  --target-tags $tag `
  --description "Temporary Qwen combiner API access"
```

If the rule already exists and the public IP changed:

```powershell
gcloud.cmd compute firewall-rules update $firewall `
  --project $project `
  --source-ranges "$publicIp/32"
```

## Instalar Driver Y Dependencias Base

```powershell
$remote = @'
set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip git curl jq ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
'@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

The SSH command usually ends with a connection error because the VM reboots. Wait and validate the GPU:

```powershell
$deadline = (Get-Date).AddMinutes(5)
do {
  Start-Sleep -Seconds 10
  gcloud.cmd compute ssh $vm `
    --zone $createdZone `
    --project $project `
    --command "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"

  if ($LASTEXITCODE -eq 0) { break }
} while ((Get-Date) -lt $deadline)
```

Expected output includes `NVIDIA L4`.

## Preparar Workspace Python

```powershell
$remote = @'
set -euo pipefail
mkdir -p ~/qwen-combiner
cd ~/qwen-combiner

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"
uv venv
source .venv/bin/activate
uv pip install torch "transformers>=4.51.0" accelerate "peft>=0.19.1" bitsandbytes safetensors fastapi "uvicorn[standard]"
'@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

## Descargar Adapter

```powershell
$remote = @'
set -euo pipefail
cd ~/qwen-combiner

gcloud storage cp --recursive \
  "gs://llm-craft-bucket/runs/2026-07-09_0641_qwen3_4b_dpo_softce/best_adapter" \
  .

ls -lh best_adapter
sha256sum best_adapter/adapter_model.safetensors
'@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

SHA esperado:

```text
f3a608751c65d6ef479be3fb4edd36c9f7eac5737f0e01d7be43e866dcdcaff8
```

## Crear API

```powershell
$server = @'
import os
import re
import time
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "best_adapter")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3-4b-dpo-softce")
API_KEY = os.environ.get("QWEN_API_KEY", "")

THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
PREFIX_RE = re.compile(r"^(resulting\s+concept|concept)\s*:\s*", re.IGNORECASE)

app = FastAPI()

tokenizer = None
model = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 0.0


def parse_concept(raw: str) -> str:
    text = SPECIAL_TOKEN_RE.sub("", THINK_RE.sub("", raw)).strip()
    for line in text.splitlines():
        cleaned = PREFIX_RE.sub("", line.strip().strip("\"'`")).strip()
        if cleaned:
            return re.sub(r"\s+", " ", cleaned).lower()
    return ""


def authorize(authorization: str | None) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


def load_model() -> None:
    global tokenizer, model

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": model is not None, "model": MODEL_NAME}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    authorize(authorization)
    return {"data": [{"id": MODEL_NAME, "object": "model"}], "object": "list"}


@app.post("/v1/chat/completions")
def chat_completions(
    body: ChatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    authorize(authorization)

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    prompt = tokenizer.apply_chat_template(
        [message.model_dump() for message in body.messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=body.temperature > 0,
            temperature=max(body.temperature, 1e-5),
            max_new_tokens=body.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=False).strip()
    parsed = parse_concept(raw)

    return {
        "id": f"qwen-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": parsed,
                    "raw_content": raw,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "latency_s": time.perf_counter() - started,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "completion_tokens": int(generated.shape[-1]),
        },
    }
'@

$serverB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($server))

$remote = @"
set -euo pipefail
cd ~/qwen-combiner
echo '$serverB64' | base64 -d > server.py
source .venv/bin/activate
python -m py_compile server.py
"@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

## Levantar API

```powershell
$remote = @"
set -euo pipefail
cd ~/qwen-combiner

if [ -f server.pid ] && kill -0 "`$(cat server.pid)" 2>/dev/null; then
  kill "`$(cat server.pid)" || true
  sleep 2
fi

source .venv/bin/activate
export QWEN_API_KEY="$apiKey"
export ADAPTER_PATH="best_adapter"
export MODEL_NAME="$modelName"
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo `$! > server.pid
cat server.pid
"@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

Wait for model load:

```powershell
$remote = @'
set +e
cd ~/qwen-combiner
for i in $(seq 1 60); do
  code=$(curl -s -o /tmp/qwen-health.json -w "%{http_code}" http://localhost:8000/health || true)
  if [ "$code" = "200" ]; then
    cat /tmp/qwen-health.json
    exit 0
  fi
  if ! kill -0 "$(cat server.pid)" 2>/dev/null; then
    echo "SERVER_EXITED"
    tail -n 120 server.log
    exit 1
  fi
  sleep 10
done
echo "TIMEOUT"
tail -n 120 server.log
exit 1
'@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

## Obtener IP Efimera

```powershell
$ip = gcloud.cmd compute instances describe $vm `
  --zone $createdZone `
  --project $project `
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"

$ip
```

## Probar API

Desde la maquina local:

```powershell
Invoke-RestMethod -Uri "http://$ip:8000/health"
```

Test de inferencia:

```powershell
$payload = @{
  model = $modelName
  messages = @(
    @{ role = "system"; content = "You combine two concepts into one resulting concept." },
    @{ role = "user"; content = "Given two concepts, combine them into one resulting concept.`n`nConcept A: fire`nConcept B: water`n`nReturn only the resulting concept." }
  )
  max_tokens = 16
  temperature = 0
}

Invoke-RestMethod `
  -Uri "http://$ip:8000/v1/chat/completions" `
  -Method Post `
  -Headers @{ Authorization = "Bearer $apiKey" } `
  -ContentType "application/json" `
  -Body ($payload | ConvertTo-Json -Depth 5) |
  ConvertTo-Json -Depth 10
```

Expected content:

```text
steam
```

## Configurar Web Local

Actualizar `apps/web/.env.local`:

```env
QWEN_COMBINER_BASE_URL=http://<IP_EFIMERA>:8000/v1
QWEN_COMBINER_API_KEY=dev-qwen-key
QWEN_COMBINER_MODEL=qwen3-4b-dpo-softce
```

Si la VM se apaga y prende, la IP efimera puede cambiar. Volver a obtener `$ip` y actualizar `.env.local`.

## Verificacion Operativa

```powershell
$remote = @'
set -euo pipefail
cd ~/qwen-combiner
printf 'pid='
cat server.pid
ps -p "$(cat server.pid)" -o pid,cmd --no-headers
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
curl -s http://localhost:8000/health
'@

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
gcloud.cmd compute ssh $vm --zone $createdZone --project $project --command "echo $b64 | base64 -d | bash"
```

## Apagar

La IP efimera puede cambiar al volver a prender.

```powershell
gcloud.cmd compute instances stop $vm --zone $createdZone --project $project
```

Prender:

```powershell
gcloud.cmd compute instances start $vm --zone $createdZone --project $project
```

Despues de prender, obtener IP nuevamente y levantar la API si no quedo corriendo.

## Borrar Al Terminar

```powershell
gcloud.cmd compute instances delete $vm --zone $createdZone --project $project
gcloud.cmd compute firewall-rules delete $firewall --project $project
```

## Notas

- No subir el adapter al repo.
- No commitear `.env.local`.
- Para uso temporal esta API corre con `nohup`; para algo durable conviene crear un servicio `systemd`.
- Si `localhost` funciona pero la IP publica falla, revisar firewall, tag `qwen-combiner` y que la IP publica actual este en `source-ranges`.
- Si la creacion falla por stock, cambiar solo la zona y mantener la misma configuracion de maquina/GPU.
