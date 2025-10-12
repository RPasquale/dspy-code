#!/bin/bash
set -euo pipefail

echo "[entrypoint] starting embeddings indexer"

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-${KAFKA_BOOTSTRAP:-kafka:9092}}"
PRIMARY_BOOTSTRAP=$(echo "$BOOTSTRAP" | cut -d',' -f1)
case "${PRIMARY_BOOTSTRAP}" in
  *:*)
    HOST_PART="${PRIMARY_BOOTSTRAP%%:*}"
    PORT_PART="${PRIMARY_BOOTSTRAP##*:}"
    ;;
  *)
    HOST_PART="${PRIMARY_BOOTSTRAP}"
    PORT_PART="9092"
    ;;
esac
export KAFKA_HOST="${HOST_PART:-kafka}"
export KAFKA_PORT="${PORT_PART:-9092}"

. /entrypoints/wait_for_kafka.sh

WORKSPACE=${DSPY_WORKSPACE:-/workspace}
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$WORKSPACE"
else
  export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
fi

exec python -m dspy_agent.embedding.kafka_indexer
