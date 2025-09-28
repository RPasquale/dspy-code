#!/bin/bash
set -e

KAFKA_HOST=${KAFKA_HOST:-kafka}
KAFKA_PORT=${KAFKA_PORT:-9092}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-60}
ATTEMPT=0

echo "[entrypoint] waiting for kafka at ${KAFKA_HOST}:${KAFKA_PORT}..."

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if timeout 5s bash -c "echo > /dev/tcp/${KAFKA_HOST}/${KAFKA_PORT}" 2>/dev/null; then
        echo "[entrypoint] kafka is ready!"
        break
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    echo "[entrypoint] waiting for kafka... (attempt $ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "[entrypoint] ERROR: kafka not available after $MAX_ATTEMPTS attempts"
    exit 1
fi

# Additional verification with kafka tools if available
if command -v kafka-topics >/dev/null 2>&1; then
    echo "[entrypoint] verifying kafka with kafka-topics..."
    timeout 10s kafka-topics --bootstrap-server ${KAFKA_HOST}:${KAFKA_PORT} --list >/dev/null 2>&1 || {
        echo "[entrypoint] WARNING: kafka-topics verification failed, but connection is available"
    }
fi

echo "[entrypoint] kafka connectivity confirmed"
