#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting embeddings indexer"

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

exec python -m dspy_agent.embedding.kafka_indexer
