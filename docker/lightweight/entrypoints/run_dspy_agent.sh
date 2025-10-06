#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting dspy-agent bootstrap and waiting for deps"

until curl -sf http://ollama:11434/api/tags >/dev/null 2>&1; do
  echo "[entrypoint] waiting for ollama..."; sleep 2
done

# Use the improved Kafka wait script
source /entrypoints/wait_for_kafka.sh

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
LOGS_DIR=${DSPY_LOGS:-${WORKSPACE}/logs}

mkdir -p "$WORKSPACE"
mkdir -p "$LOGS_DIR"
mkdir -p "$LOGS_DIR/backend" || true

# Drop leftover synthetic test noise so fresh sessions don't read stale errors
LIVE_BACKEND_LOG="$LOGS_DIR/backend/live_backend.log"
if [ -f "$LIVE_BACKEND_LOG" ]; then
  : > "$LIVE_BACKEND_LOG" || true
fi

# Ensure DSPy cache can write under workspace (root FS is read-only)
export HOME="$WORKSPACE"
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi
mkdir -p "$HOME/.dspy_cache" || true
mkdir -p "/tmp/.dspy_cache" || true

# Set explicit cache directory to avoid diskcache SQLite issues
export DSPY_CACHE_DIR="$HOME/.dspy_cache"

# Disable diskcache SQLite optimizations that cause syntax errors
export DISKCACHE_DISABLE_SQLITE_OPTIMIZATIONS=1

dspy-agent stream-topics-create --bootstrap kafka:9092 || true

AUTO=${DSPY_AUTO_START:-false}
shopt -s nocasematch || true
if [[ "$AUTO" == "1" || "$AUTO" == "true" || "$AUTO" == "yes" || "$AUTO" == "on" ]]; then
  echo "[entrypoint] auto-start enabled; launching interactive agent"
  exec dspy-agent start --workspace "$WORKSPACE" --logs "$LOGS_DIR" --approval auto --model "${MODEL_NAME:-deepseek-coder:1.3b}"
fi
shopt -u nocasematch || true

echo "[entrypoint] bootstrap complete; container idle. Exec into it to run 'dspy-agent start --workspace $WORKSPACE --model ${MODEL_NAME:-deepseek-coder:1.3b}'"
exec tail -f /dev/null
