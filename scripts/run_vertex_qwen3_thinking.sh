#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_config="configs/sft/qwen3_4b_thinking_10k_example.yaml"
default_credentials="$repo_root/nlp2026-498021-8c813796c042.json"
adc_path="${HOME}/.config/gcloud/application_default_credentials.json"

if [[ ! -f "$adc_path" && -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  if [[ -f "$default_credentials" ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$default_credentials"
    echo "[vertex-run] Using service account key: $GOOGLE_APPLICATION_CREDENTIALS"
  else
    cat <<'EOF' >&2
[vertex-run] No Application Default Credentials were found.

Set up one of these options before retrying:
  1. Run: gcloud auth application-default login
  2. Export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
EOF
    exit 1
  fi
fi

run_name="${RUN_NAME:-qwen3-4b-thinking-10k}"
config_path="$default_config"
extra_args=()

while (($# > 0)); do
  case "$1" in
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --config)
      config_path="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

echo "[vertex-run] repo_root=$repo_root"
echo "[vertex-run] run_name=$run_name"
echo "[vertex-run] config=$config_path"

log_dir="${TMPDIR:-/tmp}/llm-craft-vertex"
mkdir -p "$log_dir"
log_path="$log_dir/${run_name}_$(date +%Y%m%d_%H%M%S).log"

cd "$repo_root"
nohup uv run --group vertex python -m src.sft.vertex_submit \
  --run-name "$run_name" \
  --config "$config_path" \
  "${extra_args[@]}" \
  >"$log_path" 2>&1 < /dev/null &
job_pid=$!
disown "$job_pid" 2>/dev/null || true

echo "[vertex-run] submitter_pid=$job_pid"
echo "[vertex-run] logs=$log_path"
echo "[vertex-run] Press Ctrl+C to stop following logs without cancelling the Vertex job."

trap 'printf "\n[vertex-run] Detached from log streaming. The Vertex job is still running.\n[vertex-run] Follow logs with: tail -f %s\n" "$log_path"; exit 0' INT

tail -n +1 --pid="$job_pid" -f "$log_path"
wait "$job_pid"
