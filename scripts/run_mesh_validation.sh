#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[mesh-validation] running go unit tests"
(
  cd "$ROOT_DIR/orchestrator"
  go test ./cmd/stream_supervisor/...
)

echo "[mesh-validation] running rust unit tests"
(
  cd "$ROOT_DIR/env_runner_rs"
  cargo test
)

echo "[mesh-validation] done"
