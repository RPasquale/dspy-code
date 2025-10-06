#!/bin/sh
set -e

echo "[entrypoint] starting ollama server"

# Start ollama server in background
ollama serve &
SERVE_PID=$!

# Allow configurable startup delay before attempting pulls
STARTUP_DELAY="${OLLAMA_STARTUP_DELAY:-8}"
if [ "$STARTUP_DELAY" -gt 0 ] 2>/dev/null; then
  sleep "$STARTUP_DELAY"
fi

# Determine which models to ensure are available
RAW_MODELS="${OLLAMA_MODELS:-${OLLAMA_MODEL:-deepseek-coder:1.3b}}"
NORMALIZED_MODELS=$(printf '%s' "$RAW_MODELS" | tr ',' ' ')

for MODEL in $NORMALIZED_MODELS; do
  CLEAN_MODEL=$(printf '%s' "$MODEL" | xargs)
  if [ -z "$CLEAN_MODEL" ]; then
    continue
  fi
  echo "[entrypoint] ensuring ollama model '$CLEAN_MODEL' is available"
  ollama pull "$CLEAN_MODEL" || true
done

echo "[entrypoint] ollama ready, keeping server running"

# Wait for the server process
wait $SERVE_PID
