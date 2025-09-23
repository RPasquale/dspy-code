#!/usr/bin/env bash
set -euo pipefail

# Docker-based quick smoke for lightweight stack

WORKSPACE=${WORKSPACE:-"$(pwd)"}
echo "[stress_docker] WS=$WORKSPACE"

dspy-agent lightweight_init --workspace "$WORKSPACE" || true
dspy-agent lightweight_up || true

echo "Waiting for dashboard server..."
sleep 5
set +e
curl -fsS http://localhost:8080/api/status >/dev/null && echo "status OK" || echo "status FAIL"
curl -fsS http://localhost:8080/api/signatures >/dev/null && echo "signatures OK" || echo "signatures FAIL"
curl -fsS http://localhost:8080/api/rl/sweep/state >/dev/null && echo "sweep state OK" || echo "sweep state FAIL"
set -e

echo "Triggering a short sweep via dashboard API"
curl -fsS -X POST -H 'Content-Type: application/json' -d '{"method":"eprotein","iterations":4}' http://localhost:8080/api/rl/sweep/run || true

echo "[stress_docker] Completed basic checks"
# Optional: stop stack
# dspy-agent lightweight_down

