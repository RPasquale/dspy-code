#!/bin/bash
#SBATCH -J train_agent_methodologies
#SBATCH -N ${NODES:-1}
#SBATCH --ntasks-per-node=${GPUS:-1}
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00

set -euo pipefail

# Keep placeholders visible for template validation
# Task: ${TASK_ID} Method: ${TRAINING_METHOD}
echo "[slurm] Training method ${TRAINING_METHOD:-grpo} for task ${TASK_ID:-unknown}"

torchrun \
  --nproc_per_node "${GPUS:-1}" \
  --nnodes "${NODES:-1}" \
  rl/training/trainer_fsdp.py
