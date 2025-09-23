#!/usr/bin/env bash
set -euo pipefail

# One-button dev cycle:
# - Install dev deps
# - Run tests (non-docker)
# - Build frontend
# - Build + (re)start lightweight stack; health + smoke
# - Commit and push current branch to origin

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[dev-cycle] root=$ROOT_DIR"

# 1) Install dev dependencies (uv preferred)
if command -v uv >/dev/null 2>&1; then
  echo "[dev-cycle] installing dev dependencies via uv"
  uv sync --dev
else
  echo "[dev-cycle] uv not found; attempting pip fallback (requirements.txt if present)"
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt || true
  fi
fi

# 2) Run tests (non-docker)
echo "[dev-cycle] running E2E server tests"
if command -v uv >/dev/null 2>&1; then
  uv run pytest -q tests/test_e2e_server.py
else
  pytest -q tests/test_e2e_server.py
fi

# 3) Build frontend bundle (for local server fallback)
echo "[dev-cycle] building frontend (React)"
pushd frontend/react-dashboard >/dev/null
if command -v npm >/dev/null 2>&1; then
  npm ci || true
  npm run build || true
else
  echo "[dev-cycle] npm not found; skipping local frontend build"
fi
popd >/dev/null

# 4) Build + start lightweight stack
echo "[dev-cycle] building lightweight docker stack"
make stack-build
echo "[dev-cycle] starting stack"
make stack-up
echo "[dev-cycle] health check"
make health-check || true
echo "[dev-cycle] smoke test"
make smoke || true
echo "[dev-cycle] stack status"
make stack-ps || true

# 5) Commit + push
echo "[dev-cycle] preparing git push"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
git add -A
if ! git diff --cached --quiet; then
  msg="chore(dev-cycle): build/test/lightweight update $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  git commit -m "$msg" || true
else
  echo "[dev-cycle] no staged changes to commit"
fi
echo "[dev-cycle] pulling latest (rebase)"
git pull --rebase || true
echo "[dev-cycle] pushing to origin/$branch"
git push origin "$branch" || true

echo "[dev-cycle] done. Dashboard likely at http://127.0.0.1:18081/dashboard"
