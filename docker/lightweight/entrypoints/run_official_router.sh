#!/bin/bash
set -euo pipefail

AGENT_SERVICE=${AGENT_SERVICE:-infermesh-node-a:50051}
HTTP_PORT=${ROUTER_HTTP_PORT:-9000}
GRPC_PORT=${ROUTER_GRPC_PORT:-9100}
MAX_REQUESTS=${ROUTER_MAX_REQUESTS:-512}
REQUEST_TIMEOUT=${ROUTER_REQUEST_TIMEOUT:-30}
BIND_ADDR=${ROUTER_BIND_ADDR:-0.0.0.0}

echo "[router] waiting for mesh agent at ${AGENT_SERVICE}"

# Wait for the agent to be reachable
HOST=$(echo "$AGENT_SERVICE" | cut -d: -f1)
PORT=$(echo "$AGENT_SERVICE" | cut -d: -f2)

# Wait for the agent to be reachable using timeout
while ! timeout 1 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; do
  echo "[router] waiting for mesh agent at ${AGENT_SERVICE}"
  sleep 2
done

echo "[router] mesh agent is available, resolving IP address..."

# Resolve hostname to IP
AGENT_IP=$(getent hosts "$HOST" | awk '{ print $1 }' | head -1)

if [ -z "$AGENT_IP" ]; then
    echo "[router] ERROR: Could not resolve hostname $HOST"
    exit 1
fi

TARGET="${AGENT_IP}:${PORT}"
echo "[router] resolved agent endpoint: ${TARGET}"

exec /usr/local/bin/mesh-router \
  --bind "${BIND_ADDR}" \
  --http-port "${HTTP_PORT}" \
  --grpc-port "${GRPC_PORT}" \
  --agent "${TARGET}" \
  --max-requests "${MAX_REQUESTS}" \
  --request-timeout "${REQUEST_TIMEOUT}"
