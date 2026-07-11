#!/usr/bin/env bash
# Build and deploy apps/web to Cloud Run.
#
# Usage:
#   DATABASE_URL="postgres://..." ./scripts/gcp/deploy_web_cloudrun.sh
#
# Optional env:
#   QWEN_COMBINER_BASE_URL  e.g. http://<model-vm-ip>:8000/v1 (omit to run Gemini-only)
#   QWEN_COMBINER_API_KEY   bearer token expected by the model VM
#   VERTEX_MODEL            default gemini-2.5-flash
#   ADMIN_DASH_USER / ADMIN_DASH_PASSWORD  enable the /admin GPU dashboard
#   QWEN_VM_NAME / QWEN_VM_ZONE            VM targeted by /admin (defaults in app)
#   REGION / SERVICE / PROJECT overrides
#
# Env vars are only (re)set by this script. The Cloud Build CI trigger
# (cloudbuild.web.yaml) redeploys the image without touching env.
set -euo pipefail

PROJECT="${PROJECT:-nlp2026-498021}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-llm-craft-web}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/llm-craft-registry/${SERVICE}:latest"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required (managed Postgres reachable from Cloud Run)." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> Building image ${IMAGE}"
gcloud builds submit "${REPO_ROOT}/apps/web" \
  --project "${PROJECT}" \
  --tag "${IMAGE}"

ENV_VARS="DATABASE_URL=${DATABASE_URL}"
ENV_VARS+="@VERTEX_USE_GCE_METADATA=true"
ENV_VARS+="@GOOGLE_CLOUD_PROJECT=${PROJECT}"
ENV_VARS+="@VERTEX_LOCATION=${REGION}"
ENV_VARS+="@VERTEX_MODEL=${VERTEX_MODEL:-gemini-2.5-flash}"

if [[ -n "${QWEN_COMBINER_BASE_URL:-}" ]]; then
  ENV_VARS+="@QWEN_COMBINER_BASE_URL=${QWEN_COMBINER_BASE_URL}"
fi

if [[ -n "${QWEN_COMBINER_API_KEY:-}" ]]; then
  ENV_VARS+="@QWEN_COMBINER_API_KEY=${QWEN_COMBINER_API_KEY}"
fi

if [[ -n "${ADMIN_DASH_USER:-}" && -n "${ADMIN_DASH_PASSWORD:-}" ]]; then
  ENV_VARS+="@ADMIN_DASH_USER=${ADMIN_DASH_USER}"
  ENV_VARS+="@ADMIN_DASH_PASSWORD=${ADMIN_DASH_PASSWORD}"
  ENV_VARS+="@QWEN_VM_NAME=${QWEN_VM_NAME:-qwen-combiner-test}"
  ENV_VARS+="@QWEN_VM_ZONE=${QWEN_VM_ZONE:-us-central1-a}"
else
  echo "NOTE: ADMIN_DASH_USER/ADMIN_DASH_PASSWORD not set; /admin dashboard will be disabled." >&2
fi

echo "==> Deploying ${SERVICE} to Cloud Run (${REGION})"
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 3 \
  --memory 512Mi \
  --cpu 1 \
  --set-env-vars "^@^${ENV_VARS}"

echo "==> Done. Service URL:"
gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --format 'value(status.url)'
