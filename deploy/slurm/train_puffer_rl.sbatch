#!/bin/bash
#SBATCH -J puffer_rl_train
#SBATCH -N ${NODES:-1}
#SBATCH --ntasks-per-node=${GPUS:-1}
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=${CPUS_PER_TASK:-8}
#SBATCH --mem=${MEMORY_GB:-48}G
#SBATCH -t ${TIME_LIMIT:-04:00:00}
#SBATCH --output=${LOG_DIR:-/workspace/logs}/puffer_rl_%j.out
#SBATCH --error=${LOG_DIR:-/workspace/logs}/puffer_rl_%j.err

set -euo pipefail

TASK_ID=${TASK_ID:-unknown}
WORKSPACE=${WORKSPACE_DIR:-$SLURM_SUBMIT_DIR}
LOG_DIR=${LOG_DIR:-/workspace/logs}
STEPS=${RL_STEPS:-1000}
N_ENVS=${RL_N_ENVS:-4}
LR=${RL_LR:-0.001}
ENTROPY=${RL_ENTROPY:-0.01}
REPLAY_CAPACITY=${RL_REPLAY_CAPACITY:-4096}
REPLAY_BATCH=${RL_REPLAY_BATCH:-256}
GRAD_CLIP=${RL_GRAD_CLIP:-1.0}
CHECKPOINT_DIR=${RL_CHECKPOINT_DIR:-$WORKSPACE/logs/rl_checkpoints}
CHECKPOINT_INTERVAL=${RL_CHECKPOINT_INTERVAL:-0}
EARLY_STOP=${RL_EARLY_STOP:-0}
LOG_INTERVAL=${RL_LOG_INTERVAL:-10}
SKIP_GEPA=${RL_SKIP_GEPA:-0}
GEPA_MODULES=${RL_GEPA_MODULES:-}
ECHO_ACTIONS=${RL_ECHO_ACTIONS:-0}
LOG_JSONL=${RL_LOG_JSONL:-$WORKSPACE/logs/rl_${TASK_ID}.jsonl}
RUST_RELEASE=${RUST_RELEASE_BUILD:-1}
SKILL=${RL_SKILL_OVERRIDE:-${RL_SKILL:-general}}

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

cd "$WORKSPACE"
echo "[slurm][puffer_rl] Task=$TASK_ID workspace=$WORKSPACE steps=$STEPS n_envs=$N_ENVS"
export RL_SKILL="$SKILL"

ENV_RUNNER_ENABLED="0"
if [[ "${START_ENV_RUNNER:-1}" == "1" && -n "${MESH_ENDPOINT:-}" ]]; then
  if [[ "$RUST_RELEASE" == "1" ]]; then
    echo "[slurm][puffer_rl] Building Rust environment runner (release)"
    cargo build --release --manifest-path env_runner_rs/Cargo.toml
  fi
  echo "[slurm][puffer_rl] Starting Rust env runner against mesh ${MESH_ENDPOINT}"
  ENV_RUNNER_ENABLED="1"
  RUST_LOG=${RUST_LOG_LEVEL:-info} ./target/release/env_runner \
    --mesh-endpoint "${MESH_ENDPOINT}" \
    --mesh-node-id "${MESH_NODE_ID:-9002}" \
    --mesh-domain "${MESH_DOMAIN:-default}" \
    --supervisor-addr "${SUPERVISOR_GRPC_ADDR:-http://127.0.0.1:7000}" \
    > "$LOG_DIR/env_runner_${SLURM_JOB_ID}.log" 2>&1 &
  ENV_RUNNER_PID=$!
  echo "[slurm][puffer_rl] env_runner PID=$ENV_RUNNER_PID"
else
  echo "[slurm][puffer_rl] Mesh endpoint not provided; skipping env runner launch"
fi

trap '[[ "$ENV_RUNNER_ENABLED" == "1" && -n "${ENV_RUNNER_PID:-}" ]] && kill ${ENV_RUNNER_PID} || true' EXIT

RL_CMD=("rl" "train" "--workspace" "$WORKSPACE" "--steps" "$STEPS" "--n-envs" "$N_ENVS" \
        "--lr" "$LR" "--entropy" "$ENTROPY" "--replay-capacity" "$REPLAY_CAPACITY" \
        "--replay-batch" "$REPLAY_BATCH" "--grad-clip" "$GRAD_CLIP" \
        "--checkpoint-dir" "$CHECKPOINT_DIR" "--checkpoint-interval" "$CHECKPOINT_INTERVAL" \
        "--early-stop" "$EARLY_STOP" "--log-interval" "$LOG_INTERVAL" "--log-jsonl" "$LOG_JSONL")

if [[ -n "$SKILL" ]]; then
  RL_CMD+=("--skill" "$SKILL")
fi

if [[ "$ECHO_ACTIONS" == "1" ]]; then
  RL_CMD+=("--echo-actions")
fi

if [[ "$SKIP_GEPA" == "1" ]]; then
  RL_CMD+=("--skip-gepa")
fi

if [[ -n "$GEPA_MODULES" ]]; then
  for module in $GEPA_MODULES; do
    RL_CMD+=("--gepa-module" "$module")
  done
fi

echo "[slurm][puffer_rl] Launching: ${RL_CMD[*]}"

if command -v uv >/dev/null 2>&1; then
  echo "[slurm][puffer_rl] Using uv run"
  uv run dspy-agent "${RL_CMD[@]}"
else
  echo "[slurm][puffer_rl] Fallback to python -m dspy_agent.cli"
  python -m dspy_agent.cli "${RL_CMD[@]}"
fi

echo "[slurm][puffer_rl] Training finished"
