#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <python_script> [extra spark-submit args...]" >&2
  exit 2
fi

SCRIPT_PATH="$1"; shift || true

export PYSPARK_PYTHON=${PYSPARK_PYTHON:-python3}

spark-submit \
  --master local[*] \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  "$@" \
  "$SCRIPT_PATH"

