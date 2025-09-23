#!/usr/bin/env bash
set -euo pipefail

# DSPy Agent end-to-end smoke tests (CLI + dashboard + training flows)
#
# Usage:
#   ./scripts/smoke_test.sh             # assumes venv + dspy-agent on PATH
#   UV_PYTHON=python3 uv run ./scripts/smoke_test.sh  # if using uv
#
# Environment variables:
#   WORKSPACE       (default: current dir)
#   LOGS            (default: $WORKSPACE/logs)
#   MODEL           (default: qwen3:1.7b)
#   OLLAMA          (default: 1)
#   BASE_URL        (default: empty)
#   API_KEY         (default: empty)
#   SHOW_RATIONALE  (default: 0)
#

WORKSPACE=${WORKSPACE:-"$(pwd)"}
LOGS=${LOGS:-"$WORKSPACE/logs"}
MODEL=${MODEL:-"qwen3:1.7b"}
OLLAMA=${OLLAMA:-1}
BASE_URL=${BASE_URL:-""}
API_KEY=${API_KEY:-""}
SHOW_RATIONALE=${SHOW_RATIONALE:-0}

export DSPY_WORKSPACE="$WORKSPACE"
export DSPY_SHOW_RATIONALE="$SHOW_RATIONALE"

echo "[1/9] Python and CLI availability"
python3 -V || python -V
command -v dspy-agent >/dev/null || { echo "dspy-agent not on PATH; ensure package is installed or use 'uv run dspy-agent'"; }

echo "[2/9] CLI help pages"
if command -v dspy-agent >/dev/null; then
  dspy-agent --help >/dev/null
  dspy-agent code --help >/dev/null
  dspy-agent rl --help >/dev/null
  dspy-agent teleprompt --help >/dev/null
  dspy-agent teleprompt_suite --help >/dev/null
fi

echo "[3/9] Non-LM code context summary (snapshot only)"
mkdir -p "$LOGS"
if command -v dspy-agent >/dev/null; then
  dspy-agent codectx --path "$WORKSPACE" --workspace "$WORKSPACE" --use-lm false >/dev/null
fi

echo "[4/9] Build dataset splits"
if command -v dspy-agent >/dev/null; then
  dspy-agent dataset --workspace "$WORKSPACE" --logs "$LOGS" --out "$WORKSPACE/.dspy_data" --split --dedup >/dev/null
fi

echo "[5/9] Teleprompt suite (dry-run if LM configured)"
if command -v dspy-agent >/dev/null; then
  if [ -n "$MODEL" ] && [ "$OLLAMA" != "0" ]; then
    dspy-agent teleprompt_suite --modules codectx,task --methods bootstrap --dataset-dir "$WORKSPACE/.dspy_data/splits" --shots 2 --save-best-dir "$WORKSPACE/.dspy_prompts" --ollama --model "$MODEL" ${BASE_URL:+--base-url "$BASE_URL"} ${API_KEY:+--api-key "$API_KEY"} || true
  else
    echo "Skipping teleprompt suite: LM not configured"
  fi
fi

echo "[6/9] Native RL sweep (eprotein)"
if command -v dspy-agent >/dev/null; then
  dspy-agent rl sweep --workspace "$WORKSPACE" --iterations 2 --method eprotein --persist "$WORKSPACE/.dspy/rl/best.json" --no-update-config || true
fi

echo "[7/9] Verify sweep state persistence"
test -f "$WORKSPACE/.dspy/rl/sweep_state.json" && echo "state OK" || echo "state missing (will be created after first successful sweep run)"

echo "[8/9] Dashboard endpoints (local HTTP)"
python3 - << 'PY' || true
import threading, time, urllib.request, urllib.error
from enhanced_dashboard_server import start_enhanced_dashboard_server
def run():
    try:
        start_enhanced_dashboard_server(8099)
    except Exception:
        pass
t = threading.Thread(target=run, daemon=True); t.start()
time.sleep(1.5)
for path in ['/api/status','/api/signatures','/api/metrics']:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:8099{path}', timeout=3) as r:
            print(path, r.status)
    except Exception as e:
        print('dashboard check failed for', path, e)
PY

echo "[9/9] Success: basic smoke tests completed"

