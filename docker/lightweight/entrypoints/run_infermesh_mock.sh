#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting infermesh mock on :9000"

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi
exec python -m dspy_agent.embedding.infermesh_mock
