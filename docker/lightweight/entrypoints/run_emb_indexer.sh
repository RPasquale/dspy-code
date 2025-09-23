#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting embeddings indexer"
exec python /app/scripts/indexer.py

