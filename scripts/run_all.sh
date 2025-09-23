#!/usr/bin/env bash
set -euo pipefail

# End-to-end dev → test → build → run → (optional) publish helper
#
# Usage:
#   scripts/run_all.sh [--skip-install] [--skip-tests] [--skip-frontend] [--skip-docker]
#                      [--pypi] [--docker-push]
#
# Env:
#   ADMIN_KEY           (optional) used by server/cleanup admin gating when running stack
#   WORKSPACE_DIR       (optional) used by docker compose .env (defaults to CWD)
#   TWINE_USERNAME/__token__ and TWINE_PASSWORD or TWINE_TOKEN for PyPI publish
#   GHCR_PAT            (optional) GitHub Container Registry token for docker push
#   GHCR_IMAGE          (optional) image name to push (e.g., ghcr.io/owner/dspy-lightweight:latest)
#
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

skip_install=0
skip_tests=0
skip_frontend=0
skip_docker=0
do_pypi=0
do_docker_push=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install) skip_install=1 ; shift ;;
    --skip-tests) skip_tests=1 ; shift ;;
    --skip-frontend) skip_frontend=1 ; shift ;;
    --skip-docker) skip_docker=1 ; shift ;;
    --pypi) do_pypi=1 ; shift ;;
    --docker-push) do_docker_push=1 ; shift ;;
    -h|--help)
      sed -n '1,80p' "$0" | sed -n '1,40p'
      exit 0
      ;;
    *) echo "[run_all] unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "[run_all] root=$ROOT_DIR"

# 1) Install dev deps (uv preferred)
if [[ $skip_install -eq 0 ]]; then
  echo "[run_all] installing dev dependencies (uv)"
  if command -v uv >/dev/null 2>&1; then
    uv sync --dev
  else
    python -m pip install --upgrade pip
    pip install -r requirements.txt || true
  fi
fi

# 2) Run non-docker tests
if [[ $skip_tests -eq 0 ]]; then
  echo "[run_all] running tests (non-docker)"
  if command -v uv >/dev/null 2>&1; then
    uv run pytest -q -m "not docker"
  else
    pytest -q -m "not docker"
  fi
fi

# 3) Build frontend (optional)
if [[ $skip_frontend -eq 0 ]]; then
  echo "[run_all] building React frontend"
  pushd frontend/react-dashboard >/dev/null
  if command -v npm >/dev/null 2>&1; then
    npm ci || true
    npm run build || true
  else
    echo "[run_all] npm not found; skipping local frontend build"
  fi
  popd >/dev/null
fi

# 4) Docker lightweight stack
if [[ $skip_docker -eq 0 ]]; then
  echo "[run_all] building docker stack"
  make stack-build
  echo "[run_all] bringing up stack"
  make stack-up
  echo "[run_all] health check"
  make health-check || true
  echo "[run_all] smoke test"
  make smoke || true
  echo "[run_all] stack status"
  make stack-ps || true
fi

# 5) Optional publish to PyPI
if [[ $do_pypi -eq 1 ]]; then
  echo "[run_all] publishing to PyPI"
  if command -v uv >/dev/null 2>&1; then
    uv build || true
  else
    python -m pip install build twine
    python -m build
  fi
  if command -v twine >/dev/null 2>&1; then
    twine check dist/*
    if [[ -n "${TWINE_TOKEN:-}" ]]; then
      python -m twine upload dist/* -u __token__ -p "$TWINE_TOKEN"
    else
      python -m twine upload dist/*
    fi
  else
    echo "[run_all] twine not found; skipping upload"
  fi
fi

# 6) Optional docker push (lightweight)
if [[ $do_docker_push -eq 1 ]]; then
  echo "[run_all] pushing docker images (lightweight)"
  if [[ -z "${GHCR_IMAGE:-}" ]]; then
    echo "[run_all] set GHCR_IMAGE (e.g., ghcr.io/owner/dspy-lightweight:latest)" >&2
  else
    if [[ -n "${GHCR_PAT:-}" ]]; then
      echo "$GHCR_PAT" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin || true
    fi
    # Attempt to tag and push the lightweight image
    docker images | awk '/dspy-lightweight/ {print $1":"$2}' | while read -r IMG; do
      echo "[run_all] tagging $IMG -> $GHCR_IMAGE"
      docker tag "$IMG" "$GHCR_IMAGE" || true
      docker push "$GHCR_IMAGE" || true
    done
  fi
fi

echo "[run_all] done"

