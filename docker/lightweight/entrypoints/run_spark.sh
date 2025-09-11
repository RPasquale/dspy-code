#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting spark log stream"

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /app/scripts/streaming/spark_logs.py \
  --bootstrap kafka:9092 \
  --pattern 'logs.raw.*' \
  --checkpoint /workspace/.dspy_checkpoints/spark_logs

