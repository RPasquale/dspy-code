#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting dspy-agent and waiting for deps"

# Wait for Ollama HTTP API
until curl -sf http://ollama:11434/api/tags >/dev/null 2>&1; do
  echo "[entrypoint] waiting for ollama..."; sleep 2
done

# Wait for Kafka broker TCP port
until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
  echo "[entrypoint] waiting for kafka..."; sleep 2
done

# Create Kafka topics if they do not exist (ignore errors)
dspy-agent stream-topics-create --bootstrap kafka:9092 || true

exec dspy-agent up --workspace /workspace --db auto --status --status-port 8765
