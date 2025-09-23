#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting dspy-agent bootstrap and waiting for deps"

until curl -sf http://ollama:11434/api/tags >/dev/null 2>&1; do
  echo "[entrypoint] waiting for ollama..."; sleep 2
done

until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
  echo "[entrypoint] waiting for kafka..."; sleep 2
done

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
LOGS_DIR=${DSPY_LOGS:-${WORKSPACE}/logs}

mkdir -p "$WORKSPACE"
mkdir -p "$LOGS_DIR"

# Ensure DSPy cache can write under workspace (root FS is read-only)
export HOME="$WORKSPACE"
mkdir -p "$HOME/.dspy_cache" || true

dspy-agent stream-topics-create --bootstrap kafka:9092 || true

AUTO=${DSPY_AUTO_START:-false}
shopt -s nocasematch || true
if [[ "$AUTO" == "1" || "$AUTO" == "true" || "$AUTO" == "yes" || "$AUTO" == "on" ]]; then
  echo "[entrypoint] auto-start enabled; launching interactive agent"
  exec dspy-agent start --workspace "$WORKSPACE" --logs "$LOGS_DIR" --approval auto
fi
shopt -u nocasematch || true

echo "[entrypoint] bootstrap complete; container idle. Exec into it to run 'dspy-agent --workspace $WORKSPACE'"
exec tail -f /dev/null
