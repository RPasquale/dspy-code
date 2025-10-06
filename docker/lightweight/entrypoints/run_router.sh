#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] waiting for kafka, then starting router"

until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
  echo "[entrypoint] waiting for kafka..."; sleep 2
done

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

python - <<'PY'
from dspy_agent.agents.router_worker import RouterWorker
RouterWorker('kafka:9092').run()
PY
