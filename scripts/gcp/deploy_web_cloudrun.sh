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
#   SKIP_BUILD=1            reuse the last pushed image, only update the service
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

if [[ -z "${SKIP_BUILD:-}" ]]; then
  echo "==> Building image ${IMAGE}"
  gcloud builds submit "${REPO_ROOT}/apps/web" \
    --project "${PROJECT}" \
    --tag "${IMAGE}"
else
  echo "==> SKIP_BUILD set, reusing ${IMAGE}"
fi

# Values can contain any character (@, :, commas), so pass them via a YAML
# env file instead of --set-env-vars delimiter juggling.
ENV_FILE="$(mktemp)"
trap 'rm -f "${ENV_FILE}"' EXIT

yaml_quote() {
  printf "'%s'" "${1//\'/\'\'}"
}

{
  echo "DATABASE_URL: $(yaml_quote "${DATABASE_URL}")"
  echo "VERTEX_USE_GCE_METADATA: 'true'"
  echo "GOOGLE_CLOUD_PROJECT: $(yaml_quote "${PROJECT}")"
  echo "VERTEX_LOCATION: $(yaml_quote "${REGION}")"
  echo "VERTEX_MODEL: $(yaml_quote "${VERTEX_MODEL:-gemini-2.5-flash}")"

  if [[ -n "${QWEN_COMBINER_BASE_URL:-}" ]]; then
    echo "QWEN_COMBINER_BASE_URL: $(yaml_quote "${QWEN_COMBINER_BASE_URL}")"
  fi

  if [[ -n "${QWEN_COMBINER_API_KEY:-}" ]]; then
    echo "QWEN_COMBINER_API_KEY: $(yaml_quote "${QWEN_COMBINER_API_KEY}")"
  fi

  if [[ -n "${ADMIN_DASH_USER:-}" && -n "${ADMIN_DASH_PASSWORD:-}" ]]; then
    echo "ADMIN_DASH_USER: $(yaml_quote "${ADMIN_DASH_USER}")"
    echo "ADMIN_DASH_PASSWORD: $(yaml_quote "${ADMIN_DASH_PASSWORD}")"
    echo "QWEN_VM_NAME: $(yaml_quote "${QWEN_VM_NAME:-qwen-combiner-test}")"
    echo "QWEN_VM_ZONE: $(yaml_quote "${QWEN_VM_ZONE:-us-central1-a}")"
  fi
} > "${ENV_FILE}"

if [[ -z "${ADMIN_DASH_USER:-}" || -z "${ADMIN_DASH_PASSWORD:-}" ]]; then
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
  --env-vars-file "${ENV_FILE}"

echo "==> Done. Service URL:"
gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --format 'value(status.url)'
