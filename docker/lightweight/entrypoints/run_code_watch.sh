#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

echo "[entrypoint] starting code watcher for $WORKSPACE"

python - <<'PY'
from pathlib import Path
from dspy_agent.code_tools.code_watch import CodeWatcher

root = Path("/workspace")
watcher = CodeWatcher(root)
watcher.run()
PY
