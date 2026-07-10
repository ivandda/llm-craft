#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-${SUDO_USER:-$USER}}"
APP_HOME="$(getent passwd "$APP_USER" | cut -d: -f6)"
REPO_URL="${REPO_URL:-https://github.com/ivandda/llm-craft.git}"
REPO_BRANCH="${REPO_BRANCH:-dev}"
REPO_DIR="${REPO_DIR:-$APP_HOME/llm-craft}"
QWEN_DIR="${QWEN_DIR:-$APP_HOME/qwen-combiner}"
ADAPTER_GCS_URI="${ADAPTER_GCS_URI:-gs://llm-craft-bucket/runs/2026-07-09_0641_qwen3_4b_dpo_softce/best_adapter}"
ADAPTER_SHA256="${ADAPTER_SHA256:-f3a608751c65d6ef479be3fb4edd36c9f7eac5737f0e01d7be43e866dcdcaff8}"
QWEN_API_KEY="${QWEN_API_KEY:-dev-qwen-key}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-4b-dpo-softce}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
DATABASE_URL="${DATABASE_URL:-postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft}"
WEB_PORT="${WEB_PORT:-3000}"
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-nlp2026-498021}"
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"
VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"
VERTEX_MODEL="${VERTEX_MODEL:-gemini-2.5-flash}"
VERTEX_USE_GCE_METADATA="${VERTEX_USE_GCE_METADATA:-true}"

log() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*"
}

as_app_user() {
  sudo -H -u "$APP_USER" bash -lc "$*"
}

install_base_packages() {
  log "Installing base packages"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    gnupg \
    jq \
    nginx \
    python3-pip \
    python3-venv \
    ubuntu-drivers-common

  if ! command -v gcloud >/dev/null 2>&1; then
    log "Installing Google Cloud CLI"
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg |
      sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" |
      sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y google-cloud-cli
  fi

  if ! command -v node >/dev/null 2>&1 || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 22 ]; then
    log "Installing Node.js 22"
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
  fi

}

install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker with compose plugin is ready"
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$APP_USER" || true
    return
  fi

  log "Installing Docker Engine and compose plugin"
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg |
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    containerd.io \
    docker-buildx-plugin \
    docker-ce \
    docker-ce-cli \
    docker-compose-plugin

  sudo systemctl enable --now docker
  sudo usermod -aG docker "$APP_USER" || true
}

ensure_nvidia_driver() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver is ready"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    return
  fi

  log "Installing NVIDIA driver and rebooting. Re-run this script after SSH is back."
  sudo ubuntu-drivers autoinstall
  sudo reboot
  exit 194
}

install_uv() {
  if ! as_app_user "command -v uv >/dev/null 2>&1"; then
    log "Installing uv for $APP_USER"
    as_app_user "curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
}

sync_repo() {
  log "Syncing repo $REPO_URL branch $REPO_BRANCH"
  if [ -d "$REPO_DIR/.git" ]; then
    as_app_user "cd '$REPO_DIR' && git fetch origin && git checkout '$REPO_BRANCH' && git pull --ff-only origin '$REPO_BRANCH'"
  else
    as_app_user "git clone --branch '$REPO_BRANCH' '$REPO_URL' '$REPO_DIR'"
  fi
}

write_web_env() {
  log "Writing web runtime env"
  local env_file
  env_file="$(mktemp)"
  cat > "$env_file" <<ENV
DATABASE_URL=$DATABASE_URL
QWEN_COMBINER_BASE_URL=http://127.0.0.1:8000/v1
QWEN_COMBINER_API_KEY=$QWEN_API_KEY
QWEN_COMBINER_MODEL=$QWEN_MODEL_NAME
GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT
VERTEX_LOCATION=$VERTEX_LOCATION
VERTEX_MODEL=$VERTEX_MODEL
VERTEX_USE_GCE_METADATA=$VERTEX_USE_GCE_METADATA
ENV

  if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    printf 'GOOGLE_APPLICATION_CREDENTIALS=%s\n' "$GOOGLE_APPLICATION_CREDENTIALS" >> "$env_file"
  fi

  sudo install -o "$APP_USER" -g "$APP_USER" -m 600 "$env_file" "$REPO_DIR/apps/web/.env.local"
  rm -f "$env_file"
}

start_postgres_and_seed() {
  log "Starting Postgres and importing final-10k"
  (cd "$REPO_DIR" && sudo docker compose up -d postgres)
  sudo docker update --restart unless-stopped llm-craft-postgres >/dev/null || true

  as_app_user "cd '$REPO_DIR' && export PATH='\$HOME/.local/bin:\$PATH' && uv sync --frozen"
  as_app_user "cd '$REPO_DIR' && export PATH='\$HOME/.local/bin:\$PATH' && uv run python -m src.data.db_migrate"
  as_app_user "cd '$REPO_DIR' && export PATH='\$HOME/.local/bin:\$PATH' && uv run python -m src.data.import_final10k_to_postgres --replace-dataset final-10k"
}

write_qwen_server() {
  log "Writing Qwen API server"
  as_app_user "mkdir -p '$QWEN_DIR'"
  as_app_user "cat > '$QWEN_DIR/server.py' <<'PY'
import os
import re
import time
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = os.environ.get(\"BASE_MODEL\", \"$BASE_MODEL\")
ADAPTER_PATH = os.environ.get(\"ADAPTER_PATH\", \"best_adapter\")
MODEL_NAME = os.environ.get(\"MODEL_NAME\", \"$QWEN_MODEL_NAME\")
API_KEY = os.environ.get(\"QWEN_API_KEY\", \"\")

THINK_RE = re.compile(r\"<think>.*?</think>\", re.IGNORECASE | re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(r\"<\\|[^|]+?\\|>\")
PREFIX_RE = re.compile(r\"^(resulting\\s+concept|concept)\\s*:\\s*\", re.IGNORECASE)

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
    text = SPECIAL_TOKEN_RE.sub(\"\", THINK_RE.sub(\"\", raw)).strip()
    for line in text.splitlines():
        cleaned = PREFIX_RE.sub(\"\", line.strip().strip(\"\\\"'\`\")).strip()
        if cleaned:
            return re.sub(r\"\\s+\", \" \", cleaned).lower()
    return \"\"


def authorize(authorization: str | None) -> None:
    if not API_KEY:
        return
    if authorization != f\"Bearer {API_KEY}\":
        raise HTTPException(status_code=401, detail=\"unauthorized\")


def load_model() -> None:
    global tokenizer, model
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=\"nf4\",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map=\"auto\",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


@app.on_event(\"startup\")
def startup() -> None:
    load_model()


@app.get(\"/health\")
def health() -> dict[str, Any]:
    return {\"ok\": model is not None, \"model\": MODEL_NAME}


@app.get(\"/v1/models\")
def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    authorize(authorization)
    return {\"data\": [{\"id\": MODEL_NAME, \"object\": \"model\"}], \"object\": \"list\"}


@app.post(\"/v1/chat/completions\")
def chat_completions(
    body: ChatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    authorize(authorization)
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=\"model not loaded\")

    prompt = tokenizer.apply_chat_template(
        [message.model_dump() for message in body.messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)
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

    generated = output[0, inputs[\"input_ids\"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=False).strip()
    parsed = parse_concept(raw)

    return {
        \"id\": f\"qwen-{int(time.time() * 1000)}\",
        \"object\": \"chat.completion\",
        \"model\": MODEL_NAME,
        \"choices\": [
            {
                \"index\": 0,
                \"message\": {
                    \"role\": \"assistant\",
                    \"content\": parsed,
                    \"raw_content\": raw,
                },
                \"finish_reason\": \"stop\",
            }
        ],
        \"usage\": {
            \"latency_s\": time.perf_counter() - started,
            \"prompt_tokens\": int(inputs[\"input_ids\"].shape[-1]),
            \"completion_tokens\": int(generated.shape[-1]),
        },
    }
PY"
}

setup_qwen() {
  log "Installing Qwen runtime and adapter"
  as_app_user "mkdir -p '$QWEN_DIR' && cd '$QWEN_DIR' && export PATH='\$HOME/.local/bin:\$PATH' && uv venv"
  as_app_user "cd '$QWEN_DIR' && export PATH='\$HOME/.local/bin:\$PATH' && uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 torch"
  as_app_user "cd '$QWEN_DIR' && export PATH='\$HOME/.local/bin:\$PATH' && uv pip install --python .venv/bin/python 'transformers>=4.51.0' accelerate 'peft>=0.19.1' bitsandbytes safetensors fastapi 'uvicorn[standard]'"

  if [ ! -f "$QWEN_DIR/best_adapter/adapter_model.safetensors" ]; then
    as_app_user "cd '$QWEN_DIR' && gcloud storage cp --recursive '$ADAPTER_GCS_URI' ."
  fi

  if [ -n "$ADAPTER_SHA256" ]; then
    local actual_sha
    actual_sha="$(sha256sum "$QWEN_DIR/best_adapter/adapter_model.safetensors" | awk '{print $1}')"
    if [ "$actual_sha" != "$ADAPTER_SHA256" ]; then
      echo "Adapter SHA mismatch: expected $ADAPTER_SHA256 got $actual_sha" >&2
      exit 1
    fi
  fi

  write_qwen_server
  as_app_user "cd '$QWEN_DIR' && .venv/bin/python -m py_compile server.py"
}

install_qwen_service() {
  log "Installing qwen-combiner systemd service"
  sudo tee /etc/systemd/system/qwen-combiner.service >/dev/null <<UNIT
[Unit]
Description=Qwen combiner API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$QWEN_DIR
Environment=BASE_MODEL=$BASE_MODEL
Environment=ADAPTER_PATH=best_adapter
Environment=MODEL_NAME=$QWEN_MODEL_NAME
Environment=QWEN_API_KEY=$QWEN_API_KEY
ExecStart=$QWEN_DIR/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  sudo systemctl enable qwen-combiner
  sudo systemctl restart qwen-combiner
}

build_web() {
  log "Installing and building Next web"
  as_app_user "cd '$REPO_DIR/apps/web' && npm ci"
  as_app_user "cd '$REPO_DIR/apps/web' && npm run build"
}

install_web_service() {
  log "Installing web systemd service"
  sudo tee /etc/systemd/system/llm-craft-web.service >/dev/null <<UNIT
[Unit]
Description=llm-craft Next web
After=network-online.target docker.service qwen-combiner.service
Wants=network-online.target qwen-combiner.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$REPO_DIR/apps/web
Environment=NODE_ENV=production
EnvironmentFile=$REPO_DIR/apps/web/.env.local
ExecStart=/usr/bin/npm run start -- --hostname 127.0.0.1 --port $WEB_PORT
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  sudo systemctl enable llm-craft-web
  sudo systemctl restart llm-craft-web
}

install_nginx() {
  log "Configuring nginx reverse proxy"
  sudo tee /etc/nginx/sites-available/llm-craft >/dev/null <<'NGINX'
server {
  listen 80 default_server;
  listen [::]:80 default_server;
  server_name _;

  client_max_body_size 10m;

  location / {
    proxy_pass http://127.0.0.1:3000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_cache_bypass $http_upgrade;
  }
}
NGINX

  sudo rm -f /etc/nginx/sites-enabled/default
  sudo ln -sf /etc/nginx/sites-available/llm-craft /etc/nginx/sites-enabled/llm-craft
  sudo nginx -t
  sudo systemctl enable nginx
  sudo systemctl reload nginx
}

wait_for_services() {
  log "Waiting for Qwen health"
  local qwen_ready=0
  for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8000/health >/tmp/qwen-health.json; then
      cat /tmp/qwen-health.json
      qwen_ready=1
      break
    fi
    sleep 10
  done

  if [ "$qwen_ready" -ne 1 ]; then
    sudo journalctl -u qwen-combiner -n 120 --no-pager || true
    echo "Qwen health check timed out." >&2
    exit 1
  fi

  log "Checking web"
  curl -fsSI http://127.0.0.1/ | head -n 5 || true
}

main() {
  install_base_packages
  install_docker
  ensure_nvidia_driver
  install_uv
  sync_repo
  write_web_env
  start_postgres_and_seed
  setup_qwen
  install_qwen_service
  build_web
  install_web_service
  install_nginx
  wait_for_services

  log "All-in-one VM is ready"
  systemctl --no-pager --full status qwen-combiner llm-craft-web nginx | sed -n '1,80p' || true
}

main "$@"
