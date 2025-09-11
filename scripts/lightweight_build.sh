#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${1:-docker/lightweight/docker-compose.yml}

echo "[lightweight-build] compose: $COMPOSE_FILE"
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  echo "Run: dspy-agent lightweight_init --workspace /abs/path --logs /abs/logs" >&2
  exit 1
fi

echo "[lightweight-build] building..."
dspy-agent lightweight_build --compose "$COMPOSE_FILE"

echo "[lightweight-build] done. Use 'dspy-agent lightweight_status' to check status."

