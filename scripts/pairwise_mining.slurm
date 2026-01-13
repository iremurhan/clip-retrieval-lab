#!/bin/bash
#
# Pairwise Image Similarity Mining
# Usage: sbatch scripts/pairwise_mining.slurm [--export=MODE=mean]
#
#SBATCH --job-name=pairwise_mining
#SBATCH --output=/users/beyza.urhan/experiments/runs/slurm_logs/%j_pairwise.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/slurm_logs/%j_pairwise_err.log
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

# 1. Container Mount Check (Fail fast if mount failed)
if [ ! -d "/workspace" ]; then
    echo "FATAL: Container mount failed. /workspace not found."
    exit 1
fi

# 2. Parameters (can be overridden with --export)
MODE="${MODE:-mean}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# 3. Directory Structure
# /output -> Host: /users/beyza.urhan/experiments/runs/
JOB_DIR="/output/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

# Create slurm_logs directory if it doesn't exist
mkdir -p "/output/slurm_logs"

# 4. WandB Configuration
export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then source /env/wandb.env; fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
else
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_pairwise_${MODE}"
fi

# 5. Run Mining Script
cd /workspace
OUTPUT_PATH="datasets/coco/pairwise_mining_${MODE}.pt"

echo "Starting Mining (Mode: $MODE)..."
echo "Logs are being saved to: $JOB_DIR/mining_process.log"

# Output to both console (Slurm log) and file (Job directory)
python3 -u tools/calculate_pairwise_similarity.py \
    --config config.yaml \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --output "$OUTPUT_PATH" \
    2>&1 | tee "$JOB_DIR/mining_process.log"