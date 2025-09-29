#!/bin/bash
#SBATCH -J train_ddp
#SBATCH -N ${NODES:-2}
#SBATCH --ntasks-per-node=${GPUS:-4}
#SBATCH --gpus-per-node=${GPUS:-4}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00

set -euo pipefail

echo "[slurm] Launching DDP training for task ${TASK_ID:-unknown}"

torchrun \
  --nproc_per_node "${GPUS:-4}" \
  --nnodes "${NODES:-2}" \
  rl/training/trainer_fsdp.py

