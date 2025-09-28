#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting embed-worker (InferMesh client)"

exec python -m dspy_agent.embedding.embed_worker
