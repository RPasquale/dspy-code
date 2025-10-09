#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT_DIR/proto"
BUF_TEMPLATE="$ROOT_DIR/buf.gen.yaml"

if ! command -v buf >/dev/null 2>&1; then
  echo "[proto] buf CLI is required; install from https://docs.buf.build/installation" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/proto/bin"
buf generate --template "$BUF_TEMPLATE" --path "$PROTO_DIR"

(cd "$ROOT_DIR/orchestrator" && go fmt ./...)
