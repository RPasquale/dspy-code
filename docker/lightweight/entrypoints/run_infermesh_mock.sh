#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting infermesh mock on :9000"
exec python -m dspy_agent.embedding.infermesh_mock
