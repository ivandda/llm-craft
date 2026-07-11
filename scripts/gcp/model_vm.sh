#!/usr/bin/env bash
# Turn the Qwen model VM on/off (the expensive GPU part of the demo).
# The web app keeps working with the VM off: cached recipes are free and
# misses fall back to Gemini.
#
# Usage: ./scripts/gcp/model_vm.sh start|stop|status
set -euo pipefail

PROJECT="${PROJECT:-nlp2026-498021}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-qwen-combiner-test}"

case "${1:-}" in
  start)
    gcloud compute instances start "${VM}" --project "${PROJECT}" --zone "${ZONE}"
    echo "External IP:"
    gcloud compute instances describe "${VM}" --project "${PROJECT}" --zone "${ZONE}" \
      --format 'value(networkInterfaces[0].accessConfigs[0].natIP)'
    ;;
  stop)
    gcloud compute instances stop "${VM}" --project "${PROJECT}" --zone "${ZONE}"
    ;;
  status)
    gcloud compute instances describe "${VM}" --project "${PROJECT}" --zone "${ZONE}" \
      --format 'value(name,status,networkInterfaces[0].accessConfigs[0].natIP)'
    ;;
  *)
    echo "Usage: $0 start|stop|status" >&2
    exit 1
    ;;
esac
