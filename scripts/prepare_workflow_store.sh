#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STORE_DIR="${1:-$ROOT_DIR/data/workflows}"

mkdir -p "$STORE_DIR"

cat <<DOC
Workflow store ready at: $STORE_DIR

Use WORKFLOW_STORE_DIR=$STORE_DIR when starting the orchestrator so workflow
revisions and metadata are persisted. Existing workflow files are untouched.
DOC
