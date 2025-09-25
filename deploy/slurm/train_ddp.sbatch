#!/usr/bin/env bash
#SBATCH -J grpo-train
#SBATCH -p gpu
#SBATCH -N ${NODES:-2}
#SBATCH --gpus-per-node=${GPUS:-4}
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

CFG=${CFG:-cfg.json}

echo "[slurm] nodes=$SLURM_NNODES gpus/node=${GPUS:-4} master=$MASTER_ADDR:$MASTER_PORT cfg=$CFG"

torchrun \
  --nnodes "$SLURM_NNODES" \
  --nproc_per_node "${GPUS:-4}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  rl/training/trainer_fsdp.py --config "$CFG"

