#!/usr/bin/env bash
set -euo pipefail

# Example: spark-submit on Kubernetes with Kafka and warehouse mounts
# Required env:
#   SPARK_IMAGE, K8S_MASTER, K8S_NAMESPACE, SVC_ACCOUNT
#   KAFKA_BOOTSTRAP, CHECKPOINT_BASE, WAREHOUSE_BASE

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <python_script> [extra spark-submit args...]" >&2
  exit 2
fi

SCRIPT_PATH="$1"; shift || true

spark-submit \
  --master "${K8S_MASTER}" \
  --deploy-mode cluster \
  --name tool-pipeline \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.kubernetes.container.image="${SPARK_IMAGE}" \
  --conf spark.kubernetes.namespace="${K8S_NAMESPACE}" \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName="${SVC_ACCOUNT}" \
  --conf spark.executor.instances=4 \
  --conf spark.executor.memory=4g \
  --conf spark.executor.cores=2 \
  "$@" \
  "$SCRIPT_PATH"

