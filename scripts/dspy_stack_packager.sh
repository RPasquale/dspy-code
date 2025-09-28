#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME=$(basename "$0")
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOG_PREFIX="[packager]"

log() { printf '%s %s\n' "$LOG_PREFIX" "$*"; }
warn() { printf '%s WARNING: %s\n' "$LOG_PREFIX" "$*" >&2; }
fatal() { printf '%s ERROR: %s\n' "$LOG_PREFIX" "$*" >&2; exit 1; }

require() { command -v "$1" >/dev/null 2>&1 || fatal "$2"; }

version_gte() {
    local IFS=.
    local v1=("$1")
    local v2=("$2")
    read -ra a <<< "$1"
    read -ra b <<< "$2"
    local len=${#a[@]}
    (( ${#b[@]} > len )) && len=${#b[@]}
    for ((i=0;i<len;i++)); do
        local ai=${a[i]:-0}
        local bi=${b[i]:-0}
        if ((10#$ai < 10#$bi)); then return 1
        elif ((10#$ai > 10#$bi)); then return 0
        fi
    done
    return 0
}

check_prereqs() {
    require go "Install Go 1.22+ before running $SCRIPT_NAME."
    require cargo "Install the Rust toolchain (cargo) before running $SCRIPT_NAME."
    require tar "Install tar to generate the bundle archive."

    local gv
    gv=$(go env GOVERSION 2>/dev/null)
    gv=${gv#go}
    [[ -z "$gv" ]] && fatal "Unable to determine Go version."
    version_gte "$gv" "1.22" || fatal "Go >=1.22 required (found $gv)."

    local cv
    cv=$(cargo --version 2>/dev/null | awk '{print $2}')
    if [[ -n "$cv" ]] && ! version_gte "$cv" "1.70"; then
        warn "Rust >=1.70 recommended (found $cv)."
    fi

    if ! command -v openssl >/dev/null 2>&1; then
        warn "OpenSSL not detected; start_bundle.sh uses it to mint secure tokens."
    fi
    if ! command -v docker >/dev/null 2>&1; then
        warn "Docker not detected; the generated bundle expects it at runtime."
    fi
}

build_go() {
    log "Building Go orchestrator binary..."
    (
        cd "$ROOT_DIR/orchestrator"
        CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" \
            -o cmd/orchestrator/orchestrator ./cmd/orchestrator
    ) || fatal "Go build failed."
}

build_rust() {
    log "Building Rust env-runner binary..."
    (
        cd "$ROOT_DIR/env_runner_rs"
        cargo build --release
    ) || fatal "Cargo build failed."

    local runner="$ROOT_DIR/env_runner_rs/target/release/env_runner"
    local runner_rs="$ROOT_DIR/env_runner_rs/target/release/env_runner_rs"
    if [[ ! -f "$runner" && -f "$runner_rs" ]]; then
        cp "$runner_rs" "$runner"
        chmod +x "$runner"
        log "Copied env_runner_rs → env_runner for Docker Compose compatibility."
    fi

    [[ -f "$runner" ]] || fatal "Rust env runner binary missing at env_runner_rs/target/release/env_runner."

    log "Building Rust RedDB server binary..."
    (
        cd "$ROOT_DIR/reddb_rs"
        cargo build --release
    ) || fatal "Cargo build for RedDB failed."

    local reddb_bin="$ROOT_DIR/reddb_rs/target/release/reddb"
    [[ -f "$reddb_bin" ]] || fatal "RedDB binary missing at reddb_rs/target/release/reddb."
}

stage_bundle() {
    DIST_DIR="$ROOT_DIR/dist"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BUNDLE_NAME="dspy_stack_bundle_${TIMESTAMP}"
    BUNDLE_DIR="$DIST_DIR/$BUNDLE_NAME"
    ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.tar.gz"

    mkdir -p "$DIST_DIR"
    [[ -e "$BUNDLE_DIR" ]] && fatal "Bundle directory already exists: $BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR"

    log "Copying repository contents into bundle..."
    (
        cd "$ROOT_DIR"
        tar \
            --exclude='./dist' \
            --exclude='./logs' \
            --exclude='./tmp' \
            --exclude='./venv' \
            --exclude='./.git' \
            --exclude='./__pycache__' \
            --exclude='*.pyc' \
            --exclude='./env_runner_rs/target' \
            --exclude='./reddb_rs/target' \
            --exclude='./frontend/**/node_modules' \
            --exclude='./frontend/**/dist' \
            --exclude='./frontend/**/.vite' \
            -cf - .
    ) | tar -xf - -C "$BUNDLE_DIR"

    mkdir -p "$BUNDLE_DIR/bin"
    cp "$ROOT_DIR/orchestrator/cmd/orchestrator/orchestrator" "$BUNDLE_DIR/bin/orchestrator"
    chmod +x "$BUNDLE_DIR/bin/orchestrator"
    cp "$ROOT_DIR/env_runner_rs/target/release/env_runner" "$BUNDLE_DIR/bin/env_runner"
    chmod +x "$BUNDLE_DIR/bin/env_runner"
    cp "$ROOT_DIR/reddb_rs/target/release/reddb" "$BUNDLE_DIR/bin/reddb"
    chmod +x "$BUNDLE_DIR/bin/reddb"

    write_start_script "$BUNDLE_DIR"
    write_readme "$BUNDLE_DIR" "$BUNDLE_NAME"

    log "Creating compressed archive..."
    tar -C "$DIST_DIR" -czf "$ARCHIVE_PATH" "$BUNDLE_NAME"

    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"
    else
        warn "sha256 utilities not found; skipping checksum."
    fi

    log "Bundle ready:"
    log "  directory: $BUNDLE_DIR"
    log "  archive:   $ARCHIVE_PATH"
    [[ -f "${ARCHIVE_PATH}.sha256" ]] && log "  checksum:  ${ARCHIVE_PATH}.sha256"
}

write_start_script() {
    local bundle_dir="$1"
    cat <<'EOF' > "$bundle_dir/start_bundle.sh"
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

log()   { printf '[bundle] %s\n' "$*"; }
warn()  { printf '[bundle] WARNING: %s\n' "$*" >&2; }
fatal() { printf '[bundle] ERROR: %s\n' "$*" >&2; exit 1; }

ensure() { command -v "$1" >/dev/null 2>&1 || fatal "$2"; }

log "Validating host dependencies..."
ensure docker "Install Docker before running this bundle."

if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
else
    fatal "Docker Compose plugin (docker compose) or docker-compose is required."
fi

if ! command -v openssl >/dev/null 2>&1; then
    fatal "OpenSSL is required to generate secure tokens (install openssl)."
fi

if ! command -v curl >/dev/null 2>&1; then
    warn "curl not found; HTTP health checks will be skipped."
fi

if ! command -v sbatch >/dev/null 2>&1 || ! command -v squeue >/dev/null 2>&1 || ! command -v sacct >/dev/null 2>&1; then
    warn "Slurm CLI tools (sbatch/squeue/sacct) not all detected; GPU job submission will be disabled."
fi

ENV_FILE="$ROOT/docker/lightweight/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    REDDB_ADMIN_TOKEN="${REDDB_ADMIN_TOKEN:-$(openssl rand -hex 32)}"
    cat > "$ENV_FILE" <<EOF_ENV
WORKSPACE_DIR=$ROOT
REDDB_ADMIN_TOKEN=$REDDB_ADMIN_TOKEN
REDDB_URL=http://reddb:8080
REDDB_NAMESPACE=dspy
REDDB_TOKEN=$REDDB_ADMIN_TOKEN
DB_BACKEND=reddb
EOF_ENV
    log "Generated docker/lightweight/.env (token masked: ${REDDB_ADMIN_TOKEN:0:4}…${REDDB_ADMIN_TOKEN: -4})"
fi

mkdir -p "$ROOT/logs/env_queue/pending" "$ROOT/logs/env_queue/done"

if [[ -x "$ROOT/bin/orchestrator" ]]; then
    log "Using prebuilt Go orchestrator from bin/orchestrator."
else
    warn "Prebuilt Go orchestrator binary not found; Docker build will compile it."
fi

if [[ -x "$ROOT/bin/env_runner" ]]; then
    log "Using prebuilt Rust env runner from bin/env_runner."
else
    warn "Prebuilt Rust env runner binary not found; Docker build will compile it."
fi

if [[ -x "$ROOT/bin/reddb" ]]; then
    log "Using prebuilt RedDB server from bin/reddb."
else
    warn "Prebuilt RedDB binary not found; Docker build will compile it."
fi

log "Building Docker images (this can take a few minutes)…"
"${COMPOSE_CMD[@]}" -f "$ROOT/docker/lightweight/docker-compose.yml" --env-file "$ENV_FILE" build

log "Starting DSPy stack…"
"${COMPOSE_CMD[@]}" -f "$ROOT/docker/lightweight/docker-compose.yml" --env-file "$ENV_FILE" up -d --remove-orphans

if command -v curl >/dev/null 2>&1; then
    log "Running quick health checks…"
    curl -fsS http://127.0.0.1:9097/metrics >/dev/null 2>&1 \
        && log "Go orchestrator responded on :9097/metrics." \
        || warn "Go orchestrator metrics endpoint not reachable yet."

    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 \
        && log "Rust env runner health endpoint responded on :8080/health." \
        || warn "Env runner health endpoint not reachable yet."
fi

log "Stack is running. Useful follow-up commands:"
log "  ${COMPOSE_CMD[*]} -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps"
log "  ${COMPOSE_CMD[*]} -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env logs -f"
log "  make -C \"$ROOT\" health-check"
log "Slurm batch templates are under deploy/slurm/."
log "To stop everything: ${COMPOSE_CMD[*]} -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env down"
EOF
    chmod +x "$bundle_dir/start_bundle.sh"
}

write_readme() {
    local bundle_dir="$1"
    local bundle_name="$2"
    local archive_name="${bundle_name}.tar.gz"
    local build_date
    build_date=$(date -u +"%Y-%m-%d %H:%M:%SZ")
    local commit_sha
    if command -v git >/dev/null 2>&1; then
        commit_sha=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    else
        commit_sha="unknown"
    fi

    cat <<EOF > "$bundle_dir/README_BUNDLE.md"
# DSPy Unified Stack Bundle

- Generated: $build_date
- Git commit: $commit_sha
- Bundle directory: $bundle_name
- Archive: $archive_name

## Host Dependencies

- Docker 20+
- Docker Compose plugin (\`docker compose\`) or legacy \`docker-compose\`
- OpenSSL (used to mint secure RedDB tokens)
- curl (optional: health checks)
- Slurm CLI tools (\`sbatch\`, \`squeue\`, \`sacct\`) for GPU job submission (optional)

## Quick Start

1. Extract the archive: \`tar -xzf $archive_name\`
2. Change into the bundle: \`cd $bundle_name\`
3. Run the unified bootstrap: \`./start_bundle.sh\`
4. Inspect services: \`${COMPOSE_CMD:-docker compose} -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env ps\`

## Contents

- \`bin/\`: Pre-built \`orchestrator\` and \`env_runner\` binaries for direct invocation.
- \`orchestrator/\`, \`env_runner_rs/\`: Source plus compiled artifacts used by the stack.
- \`deploy/slurm/\`: Slurm sbatch templates consumed by the orchestrator's bridge.
- \`docker/lightweight/\`: Docker Compose stack, entrypoints, and Dockerfile.
- \`scripts/\` & top-level startup scripts: retained for advanced workflows.
- \`start_bundle.sh\`: Single-command bootstrap that checks deps, prepares env vars, builds images, and launches the stack.

## Stopping & Maintenance

- Stop services: \`${COMPOSE_CMD:-docker compose} -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env down\`
- Tail logs: \`${COMPOSE_CMD:-docker compose} -f docker/lightweight/docker-compose.yml --env-file docker/lightweight/.env logs -f\`
- Run health checks: \`make -C . health-check\`

Keep the generated \`.env\` file private; rename/regenerate tokens as needed.
EOF
}

main() {
    check_prereqs
    build_go
    build_rust
    stage_bundle
}

main "$@"
