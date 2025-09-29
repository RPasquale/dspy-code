#!/usr/bin/env bash
set -euo pipefail

topic="${1:-app}"
echo "[entrypoint] starting dspy-agent worker for topic: ${topic}"

# Ensure DSPy disk cache lands on a writable filesystem.
# Containers run with read-only root; only volumes and tmpfs are writable.
export DSPY_WORKSPACE="${DSPY_WORKSPACE:-/workspace}"
export HOME="$DSPY_WORKSPACE"

# Create cache directories with proper permissions
mkdir -p "$HOME/.dspy_cache" || true
mkdir -p "/tmp/.dspy_cache" || true

# Set explicit cache directory to avoid diskcache SQLite issues
export DSPY_CACHE_DIR="$HOME/.dspy_cache"

# Disable diskcache SQLite optimizations that cause syntax errors
export DISKCACHE_DISABLE_SQLITE_OPTIMIZATIONS=1

# Use the improved Kafka wait script
source /entrypoints/wait_for_kafka.sh

exec dspy-agent worker --topic "${topic}" --bootstrap kafka:9092
