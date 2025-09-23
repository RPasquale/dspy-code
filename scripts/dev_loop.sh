#!/usr/bin/env bash
set -euo pipefail

CFILE="docker/lightweight/docker-compose.yml"
EFILE="docker/lightweight/.env"
AGENT_SERVICES=(dspy-agent dspy-worker dspy-worker-backend dspy-worker-frontend)

if [ ! -f "$CFILE" ]; then
  echo "[dev-loop] compose file missing: $CFILE" >&2
  exit 1
fi
if [ ! -f "$EFILE" ]; then
  echo "[dev-loop] env file missing: $EFILE — run 'make stack-env' first" >&2
  exit 1
fi

compute_sig() {
  # Hash contents of tracked files in key dirs to detect changes
  local files
  files=$(git ls-files \
    dspy_agent \
    docker/lightweight/dspy_agent \
    enhanced_dashboard_server.py \
    docker/lightweight/entrypoints \
    docker/lightweight/scripts \
    docker/lightweight/Dockerfile \
    docker/lightweight/docker-compose.yml \
    Makefile 2>/dev/null | tr '\n' ' ')
  if [ -z "$files" ]; then
    echo "none"
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    cat $files | shasum | shasum | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    cat $files | md5 | awk '{print $1}'
  else
    python3 - "$files" << 'PY'
import hashlib, sys
h=hashlib.sha256()
for f in sys.argv[1:]:
    try:
        with open(f,'rb') as fh:
            h.update(fh.read())
    except Exception:
        pass
print(h.hexdigest())
PY
  fi
}

logs_pid=""
start_logs() {
  echo "[dev-loop] tailing logs: ${AGENT_SERVICES[*]}"
  docker compose -f "$CFILE" --env-file "$EFILE" logs -f "${AGENT_SERVICES[@]}" &
  logs_pid=$!
}

stop_logs() {
  if [ -n "${logs_pid:-}" ] && kill -0 "$logs_pid" >/dev/null 2>&1; then
    echo "[dev-loop] stopping logs (pid=$logs_pid)"
    kill "$logs_pid" >/dev/null 2>&1 || true
    wait "$logs_pid" 2>/dev/null || true
  fi
}

cleanup() {
  stop_logs
}
trap cleanup EXIT INT TERM

echo "[dev-loop] initial reload"
make stack-reload
start_logs

prev="$(compute_sig)"
echo "[dev-loop] watching for changes..."
while true; do
  sleep 1
  cur="$(compute_sig)"
  if [ "$cur" != "$prev" ]; then
    echo "[dev-loop] change detected → reload"
    stop_logs
    make stack-reload || true
    start_logs
    prev="$cur"
  fi
done

