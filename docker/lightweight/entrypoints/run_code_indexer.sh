#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

BOOTSTRAP=${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}
TOPIC=${CODE_WATCH_TOPIC:-code.fs.events}

echo "[entrypoint] starting code indexer (bootstrap=$BOOTSTRAP, topic=$TOPIC)"

python - <<'PY'
import os
from pathlib import Path
from dspy_agent.code_tools.code_indexer_worker import CodeIndexerWorker

bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
topic = os.getenv('CODE_WATCH_TOPIC', 'code.fs.events')
workspace = Path(os.getenv('DSPY_WORKSPACE', '/workspace'))

worker = CodeIndexerWorker(bootstrap, topic=topic, workspace=workspace)
worker.run()
PY
