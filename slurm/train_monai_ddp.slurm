#!/bin/bash
# Example SLURM launch for a single-node, multi-GPU MONAI run.
# Adapt account/partition/module lines for your cluster.

#SBATCH --job-name=monai-demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

mkdir -p logs artifacts/run

module purge
# module load cuda/12.1
# module load python/3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

torchrun \
  --standalone \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
  train_segmentation_demo.py \
  --distributed \
  --epochs 5 \
  --batch-size 2 \
  --num-workers 4 \
  --output-dir artifacts/run/slurm
