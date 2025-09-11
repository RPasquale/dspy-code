#!/usr/bin/env bash
set -euo pipefail

topic="${1:-app}"
echo "[entrypoint] starting dspy-agent worker for topic: ${topic}"

until (echo > /dev/tcp/kafka/9092) >/dev/null 2>&1; do
  echo "[entrypoint] waiting for kafka..."; sleep 2
done

exec dspy-agent worker --topic "${topic}" --bootstrap kafka:9092
