#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting embeddings indexer"
exec python -m dspy_agent.embedding.kafka_indexer
