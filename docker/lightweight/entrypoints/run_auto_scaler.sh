#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE_DIR:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

BOOTSTRAP=${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}
REDDB_URL=${REDDB_URL:-http://reddb:8080}

echo "[entrypoint] starting auto scaler (bootstrap=$BOOTSTRAP, reddb=$REDDB_URL)"

python - <<'PY'
import os
import time
from pathlib import Path
from dspy_agent.monitor.auto_scaler import AutoScaler

workspace = Path(os.getenv('WORKSPACE_DIR', '/workspace'))
bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
reddb = os.getenv('REDDB_URL', 'http://reddb:8080')

print("[auto-scaler] initializing AutoScaler", flush=True)

scaler = AutoScaler(workspace, kafka_bootstrap=bootstrap, reddb_url=reddb)
scaler.start_monitoring()

print("[auto-scaler] monitoring loop started", flush=True)

while True:
    time.sleep(60)
PY
