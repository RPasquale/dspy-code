#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_ID="${1:-}"
API_BASE="${API_BASE:-http://localhost:9097}"

if [[ -z "$WORKFLOW_ID" ]]; then
  echo "usage: $(basename "$0") <workflow-id> [-- jq filter opts...]" >&2
  exit 1
fi

curl -sS "${API_BASE}/workflows/${WORKFLOW_ID}/history" | jq .
