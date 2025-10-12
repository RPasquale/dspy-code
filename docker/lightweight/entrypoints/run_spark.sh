#!/bin/bash
set -euo pipefail

echo "[entrypoint] starting spark log stream"

# Set up user environment to fix Hadoop authentication issues
export USER="${USER:-spark}"
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-spark}"
export SPARK_USER="${SPARK_USER:-spark}"

# Ensure Spark and Ivy use absolute directories that exist at runtime
export HOME="${HOME:-/tmp}"
export SPARK_HOME="${SPARK_HOME:-/opt/spark}"
export IVY_HOME="$HOME/.ivy2"
mkdir -p "$IVY_HOME/local"

# Guarantee JVM resolves user.home/ivy paths correctly for dependency downloads
# Also disable Hadoop security to prevent authentication issues in Docker
# Force JVM user.name to avoid JAAS UnixLoginModule NPE on null name
JAVA_PROPS="-Duser.home=$HOME -Divy.home=$IVY_HOME -Divy.default.ivy.user.dir=$IVY_HOME -Divy.cache.dir=$IVY_HOME/cache -Dhadoop.security.authentication=simple -Dhadoop.security.authorization=false -Duser.name=${USER}"
if [ -n "${JAVA_TOOL_OPTIONS:-}" ]; then
  export JAVA_TOOL_OPTIONS="$JAVA_TOOL_OPTIONS $JAVA_PROPS"
else
  export JAVA_TOOL_OPTIONS="$JAVA_PROPS"
fi

if [ -n "${SPARK_SUBMIT_OPTS:-}" ]; then
  export SPARK_SUBMIT_OPTS="$SPARK_SUBMIT_OPTS $JAVA_PROPS"
else
  export SPARK_SUBMIT_OPTS="$JAVA_PROPS"
fi

CHECKPOINT_DIR=/workspace/.dspy_checkpoints/spark_logs
mkdir -p "$(dirname "$CHECKPOINT_DIR")" "$CHECKPOINT_DIR"

# Some Hadoop libs expect the current UID to resolve to a passwd entry.
# Create a minimal passwd entry if missing (common in scratch-like images).
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  echo "spark:x:$(id -u):$(id -g):Spark User:$HOME:/bin/sh" >> /etc/passwd || true
fi

BOOTSTRAP_HOST="${SPARK_KAFKA_HOST:-kafka}"
BOOTSTRAP_PORT="${SPARK_KAFKA_PORT:-9092}"
MAX_WAIT_SECONDS="${SPARK_KAFKA_MAX_WAIT:-120}"
SLEEP_SECONDS=2
elapsed=0
until getent hosts "$BOOTSTRAP_HOST" >/dev/null 2>&1 && timeout 3 bash -c "cat < /dev/null > /dev/tcp/$BOOTSTRAP_HOST/$BOOTSTRAP_PORT" 2>/dev/null; do
  if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "[entrypoint] ERROR: Unable to reach Kafka at ${BOOTSTRAP_HOST}:${BOOTSTRAP_PORT} after ${MAX_WAIT_SECONDS}s" >&2
    exit 1
  fi
  echo "[entrypoint] Waiting for Kafka at ${BOOTSTRAP_HOST}:${BOOTSTRAP_PORT}..."
  sleep "$SLEEP_SECONDS"
  elapsed=$((elapsed + SLEEP_SECONDS))
done

exec ${SPARK_HOME}/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /app/dspy_agent/streaming/spark_logs.py \
  --bootstrap "${BOOTSTRAP_HOST}:${BOOTSTRAP_PORT}" \
  --pattern 'logs.raw.*' \
  --checkpoint "$CHECKPOINT_DIR"
