#!/bin/bash
set -euo pipefail

# Start InferMesh stack with Redis cache, dual nodes, official router, and DSPy embedder service

echo "🚀 Starting InferMesh stack deployment..."

WORKSPACE_DIR=${WORKSPACE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
INFERMESH_HOST_PORT=${INFERMESH_HOST_PORT:-19000}
EMBED_HOST_PORT=${EMBED_HOST_PORT:-18082}
INFERMESH_MODEL=${INFERMESH_MODEL:-BAAI/bge-small-en-v1.5}
INFERMESH_BATCH_SIZE=${INFERMESH_BATCH_SIZE:-512}
INFERMESH_MAX_CONCURRENCY=${INFERMESH_MAX_CONCURRENCY:-128}

cat <<CFG
📋 Configuration:
  - Workspace: $WORKSPACE_DIR
  - Router Port: $INFERMESH_HOST_PORT
  - Embedder Port: $EMBED_HOST_PORT
  - Model: $INFERMESH_MODEL
  - Batch Size: $INFERMESH_BATCH_SIZE
  - Max Concurrency: $INFERMESH_MAX_CONCURRENCY
CFG

cd "$WORKSPACE_DIR/docker/lightweight"

start_service() {
  local service="$1"
  echo "🔧 Starting $service..."
  docker compose up -d "$service"
}

wait_http() {
  local name="$1" url="$2" timeout="${3:-60}"
  echo "⏳ Waiting for $name to become healthy ($url)..."
  while [ $timeout -gt 0 ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ $name is reachable"
      return 0
    fi
    sleep 2
    timeout=$((timeout - 2))
  done
  echo "✗ $name failed to respond within the timeout"
  return 1
}

# Start Redis
start_service redis
echo "⏳ Waiting for Redis to accept connections..."
redis_timeout=40
while [ $redis_timeout -gt 0 ]; do
  if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "✓ Redis is responding"
    break
  fi
  sleep 2
  redis_timeout=$((redis_timeout - 2))
done
if [ $redis_timeout -le 0 ]; then
  echo "✗ Redis did not respond in time"
  exit 1
fi

# Start DSPy embedder first so router fallbacks work
start_service dspy-embedder
wait_http "DSPy embedder" "http://localhost:$EMBED_HOST_PORT/health" 60

# Start mesh nodes
start_service infermesh-node-a
start_service infermesh-node-b
wait_http "InferMesh node A" "http://localhost:19001/health" 60 || true
wait_http "InferMesh node B" "http://localhost:19002/health" 60 || true

# Start router
start_service infermesh-router
wait_http "InferMesh router" "http://localhost:$INFERMESH_HOST_PORT/health" 60

# Smoke test embed route through router
payload=$(cat <<JSON
{"model": "$INFERMESH_MODEL", "inputs": ["test embedding"], "options": {"batch_size": $INFERMESH_BATCH_SIZE}}
JSON
)

echo "🧪 Testing router embed endpoint..."
if curl -fsS -X POST "http://localhost:$INFERMESH_HOST_PORT/embed" \
    -H "Content-Type: application/json" \
    -d "$payload" >/dev/null; then
  echo "✓ Router embedding test succeeded"
else
  echo "✗ Router embedding test failed"
  exit 1
fi

echo ""
echo "🎉 InferMesh stack deployment successful!"
cat <<NEXT

📊 Service status hints:
  - Embedder metrics: http://localhost:$EMBED_HOST_PORT/metrics
  - Router health:    http://localhost:$INFERMESH_HOST_PORT/health

🔧 Management commands:
  - View logs:   docker compose logs -f infermesh-router
  - Check ps:    docker compose ps
  - Stop stack:  docker compose down
NEXT
