#!/usr/bin/env bash
set -euo pipefail

topic="${1:-app}"
echo "[entrypoint] starting dspy-agent worker for topic: ${topic}"

# Ensure DSPy disk cache lands on a writable filesystem.
# Containers run with read-only root; only volumes and tmpfs are writable.
export DSPY_WORKSPACE="${DSPY_WORKSPACE:-/workspace}"
export HOME="$DSPY_WORKSPACE"
mkdir -p "$HOME/.dspy_cache" || true

until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
  echo "[entrypoint] waiting for kafka..."; sleep 2
done

exec dspy-agent worker --topic "${topic}" --bootstrap kafka:9092
