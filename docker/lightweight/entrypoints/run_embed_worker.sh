#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting embed-worker (InferMesh client)"

exec python /app/scripts/embed_worker.py

