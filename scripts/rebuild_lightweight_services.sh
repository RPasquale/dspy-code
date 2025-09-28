#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker/lightweight/docker-compose.yml"
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "compose file not found: $COMPOSE_FILE" >&2
  echo "run from the repo root or pass a custom compose file path." >&2
  exit 1
fi

DEFAULT_SERVICES=(
  dspy-agent
  go-orchestrator
  rust-env-runner
  dspy-embedder
  embed-worker
  infermesh-node-a
  infermesh-node-b
  infermesh-router
  spark
  spark-vectorizer
)

if [[ $# -gt 0 ]]; then
  SERVICES=("$@")
else
  SERVICES=("${DEFAULT_SERVICES[@]}")
fi

echo "[rebuild] compose file: $COMPOSE_FILE"
echo "[rebuild] services: ${SERVICES[*]}"

echo "[rebuild] building images..."
docker compose -f "$COMPOSE_FILE" build "${SERVICES[@]}"

echo "[rebuild] restarting containers..."
docker compose -f "$COMPOSE_FILE" up -d "${SERVICES[@]}"

echo "[rebuild] current status"
docker compose -f "$COMPOSE_FILE" ps "${SERVICES[@]}"
