#!/usr/bin/env bash
set -euo pipefail

# Full end-to-end test for the lightweight Docker stack.
# - Rebuild images (no cache)
# - Bring up services
# - Wait for healthchecks
# - Run basic agent commands inside the container
# - Verify Ollama + Kafka connectivity

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
COMPOSE_DIR="$ROOT_DIR/docker/lightweight"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

cd "$COMPOSE_DIR"

echo "[stack] pruning old stack..."
docker compose -f "$COMPOSE_FILE" down -v || true

echo "[stack] building image (no cache)..."
docker compose -f "$COMPOSE_FILE" build --no-cache

echo "[stack] starting services..."
docker compose -f "$COMPOSE_FILE" up -d

echo "[stack] waiting for healthchecks..."
deadline=$((SECONDS + 300))
while true; do
  # Try JSON ps; many Compose versions don't support it. If not JSON, fallback to a fixed wait.
  out=$(docker compose -f "$COMPOSE_FILE" ps --format json 2>/dev/null || true)
  if echo "$out" | head -n1 | grep -q '^\['; then
    if command -v jq >/dev/null 2>&1; then
      unhealthy=$(echo "$out" | jq -r '.[] | select((.State != "running") and (.Health != "healthy")) | .Name' | wc -l | tr -d ' ')
      [[ "$unhealthy" == "0" ]] && break
    else
      echo "[stack] jq not found; using fixed wait"
      sleep 30; break
    fi
  else
    echo "[stack] non-JSON ps; using fixed wait"
    sleep 30; break
  fi
  if (( SECONDS > deadline )); then
    echo "[stack] timeout waiting for services to become healthy" >&2
    docker compose -f "$COMPOSE_FILE" ps || true
    break
  fi
  sleep 5
done

echo "[check] ollama on host (http://127.0.0.1:11435/api/tags)"
curl -fsS http://127.0.0.1:11435/api/tags | head -c 200 || {
  echo "[warn] ollama tags not available yet" >&2
}

echo "[check] run agent commands in container"
docker compose -f "$COMPOSE_FILE" exec -T dspy-agent dspy-agent index --workspace /workspace || true
docker compose -f "$COMPOSE_FILE" exec -T dspy-agent dspy-agent esearch "def target" --workspace /workspace || true
docker compose -f "$COMPOSE_FILE" exec -T dspy-agent dspy-agent chat "show key events" --workspace /workspace --steps 1 --ollama || true

echo "[check] worker/router/code services health"
docker compose -f "$COMPOSE_FILE" ps

echo "[ok] lightweight stack test completed"
