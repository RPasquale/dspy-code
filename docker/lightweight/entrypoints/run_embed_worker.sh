#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting embed-worker (InferMesh client)"

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

exec python -m dspy_agent.embedding.embed_worker
