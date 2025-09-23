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

# Compose config auto-fix (environment list -> mapping) if needed
COMPOSE_ORIG="$ROOT_DIR/docker/lightweight/docker-compose.yml"
STACK_ENV_FILE="$ROOT_DIR/docker/lightweight/.env"
COMPOSE_USE="$COMPOSE_ORIG"

ensure_compose_env_mapping() {
  # Ensure .env exists
  make stack-env >/dev/null 2>&1 || true

  # Prefer docker compose (v2), fallback to docker-compose
  local COMPOSE_BIN="docker compose"
  if ! docker compose version >/dev/null 2>&1; then
    if command -v docker-compose >/dev/null 2>&1; then
      COMPOSE_BIN="docker-compose"
    fi
  fi

  echo "[dev-cycle] validating compose config ($COMPOSE_BIN)"
  if $COMPOSE_BIN -f "$COMPOSE_ORIG" --env-file "$STACK_ENV_FILE" config >/dev/null 2>"$ROOT_DIR/.compose_config_err"; then
    echo "[dev-cycle] compose config OK"
    return 0
  fi

  if grep -qi "environment.*must be a mapping" "$ROOT_DIR/.compose_config_err"; then
    echo "[dev-cycle] rewriting environment blocks to mapping syntax for compatibility"
    local MAPPED="$ROOT_DIR/docker/lightweight/docker-compose.mapped.yml"
    awk '
      function startswith(str, prefix){return index(str, prefix)==1}
      function ltrim(s){sub(/^\s+/, "", s); return s}
      function rtrim(s){sub(/\s+$/, "", s); return s}
      function trim(s){return rtrim(ltrim(s))}
      BEGIN{env_mode=0; env_indent=0}
      {
        line=$0
        # Detect environment: line
        if (env_mode==0) {
          if (match(line, /^(\s*)environment:\s*$/ , m)) {
            print line
            env_mode=1
            env_indent=length(m[1])
            next
          } else {
            print line
            next
          }
        } else {
          # In environment block; expect either list items or end of block
          # A list item looks like: <indent+2>- <KEY>[=|: <VAL>]
          item_indent = env_indent + 2
          # Compute actual leading spaces
          leading=0
          for (i=1; i<=length(line); i++) { if (substr(line,i,1)!=" ") break; leading++ }
          if (leading < item_indent || !startswith(substr(line,1+item_indent), "- ")) {
            # End of env block; emit this line and exit env mode
            env_mode=0
            print line
            next
          }
          # Parse the token after "- "
          token = substr(line, item_indent+3)
          token = trim(token)
          # Cases: KEY=VAL, KEY: VAL, KEY, KEY=
          if (match(token, /^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/, mm)) {
            key=mm[1]; val=mm[2]
            gsub(/"/, "\\\"", val)
            printf "%*s%s: \"%s\"\n", item_indent, "", key, val
            next
          }
          if (match(token, /^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$/, mm2)) {
            key=mm2[1]; val=mm2[2]
            # Quote value if contains colon or spaces and not already quoted
            if (val ~ /[:\s]/ && val !~ /^\".*\"$/) {
              gsub(/"/, "\\\"", val)
              printf "%*s%s: \"%s\"\n", item_indent, "", key, val
            } else {
              printf "%*s%s: %s\n", item_indent, "", key, val
            }
            next
          }
          if (match(token, /^([A-Za-z_][A-Za-z0-9_]*)(=)?$/, mm3)) {
            key=mm3[1]
            # Preserve pass-through semantics using ${VAR:-}
            printf "%*s%s: ${%s:-}\n", item_indent, "", key, key
            next
          }
          # Fallback: print as-is (should not happen)
          print line
          next
        }
      }
    ' "$COMPOSE_ORIG" > "$MAPPED"

    if $COMPOSE_BIN -f "$MAPPED" --env-file "$STACK_ENV_FILE" config >/dev/null 2>&1; then
      echo "[dev-cycle] using mapped compose file: $MAPPED"
      COMPOSE_USE="$MAPPED"
      export STACK_COMPOSE="$MAPPED"
      return 0
    else
      echo "[dev-cycle] mapped compose still invalid; see $ROOT_DIR/.compose_config_err"
      echo "[dev-cycle] remediation: update Compose to v2.x or convert environment lists to key: value maps."
      # Fall back to original; downstream may fail, but at least continue
      return 1
    fi
  else
    echo "[dev-cycle] compose config error (non-environment); see $ROOT_DIR/.compose_config_err"
    return 1
  fi
}

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
if [[ -d frontend/react-dashboard ]]; then
  pushd frontend/react-dashboard >/dev/null
  if command -v npm >/dev/null 2>&1; then
    npm ci || true
    npm run build || true
  else
    echo "[dev-cycle] npm not found; skipping local frontend build"
  fi
  popd >/dev/null || true
else
  echo "[dev-cycle] no frontend/react-dashboard directory; skipping frontend build"
fi

# 4) Build + start lightweight stack (with compose validation/auto-fix)
ensure_compose_env_mapping || true
echo "[dev-cycle] building lightweight docker stack"
STACK_COMPOSE="${STACK_COMPOSE:-$COMPOSE_USE}" make stack-build
echo "[dev-cycle] starting stack"
STACK_COMPOSE="${STACK_COMPOSE:-$COMPOSE_USE}" make stack-up
echo "[dev-cycle] health check"
make health-check || true
echo "[dev-cycle] smoke test"
STACK_COMPOSE="${STACK_COMPOSE:-$COMPOSE_USE}" make smoke || true
echo "[dev-cycle] stack status"
STACK_COMPOSE="${STACK_COMPOSE:-$COMPOSE_USE}" make stack-ps || true

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
