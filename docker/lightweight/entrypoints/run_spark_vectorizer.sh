#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting spark vectorizer"

export USER="${USER:-spark}"
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-spark}"
export SPARK_USER="${SPARK_USER:-spark}"
export HOME="/tmp"
export SPARK_HOME=/opt/bitnami/spark
export IVY_HOME="$HOME/.ivy2"
mkdir -p "$IVY_HOME/local" || true

JAVA_PROPS="-Duser.home=$HOME -Divy.home=$IVY_HOME -Divy.default.ivy.user.dir=$IVY_HOME -Divy.cache.dir=$IVY_HOME/cache -Dhadoop.security.authentication=simple -Dhadoop.security.authorization=false -Duser.name=${USER}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} $JAVA_PROPS"
export SPARK_SUBMIT_OPTS="${SPARK_SUBMIT_OPTS:-} $JAVA_PROPS"

CHECKPOINT_DIR=${VEC_CHECKPOINT:-/workspace/.dspy_checkpoints/vectorizer}
mkdir -p "$CHECKPOINT_DIR"

if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  echo "spark:x:$(id -u):$(id -g):Spark User:$HOME:/bin/sh" >> /etc/passwd || true
fi

# Add startup delay to ensure Kafka is fully ready
STARTUP_DELAY=${SPARK_STARTUP_DELAY:-10}
echo "[entrypoint] waiting ${STARTUP_DELAY} seconds for Kafka to be fully ready..."
sleep "$STARTUP_DELAY"

echo "[entrypoint] starting spark vectorizer with enhanced timeout configuration"

exec "$SPARK_HOME/bin/spark-submit" \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /app/scripts/streaming/spark_vectorize.py
