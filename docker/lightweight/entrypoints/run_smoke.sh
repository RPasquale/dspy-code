#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] running smoke test for embeddings pipeline"

export KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-kafka:9092}"
export RESULTS_TOPIC="${RESULTS_TOPIC:-agent.results}"
export EMBED_PARQUET_DIR="${EMBED_PARQUET_DIR:-/workspace/vectorized/embeddings_imesh}"
export N_MESSAGES="${N_MESSAGES:-5}"
export SLEEP_SEC="${SLEEP_SEC:-5}"

python /app/scripts/smoke_embed_pipeline.py

