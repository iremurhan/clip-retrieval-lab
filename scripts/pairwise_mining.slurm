#!/bin/bash
#
# Pairwise Image Similarity Mining for Knowledge Distillation
#
# Usage: sbatch scripts/pairwise_mining.slurm [--export=MODE=mean]
#        Default mode: mean (consensus)
#
#SBATCH --job-name=pairwise_mining
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j_pairwise.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j_pairwise_err.log
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING (Same mounts as train.slurm)
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env

# W&B Key (for login)
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

# ------------------------------------------------------------------
# DEFAULT PARAMETERS (can be overridden with --export)
# ------------------------------------------------------------------
MODE="${MODE:-mean}"        # Aggregation mode: mean, max, min
BATCH_SIZE="${BATCH_SIZE:-512}"

# ------------------------------------------------------------------
# JOB DIRECTORIES
# ------------------------------------------------------------------
JOB_DIR="/output/${SLURM_JOB_ID}"
LOGS_DIR="$JOB_DIR/logs"

mkdir -p "$JOB_DIR" "$LOGS_DIR"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Task:        Pairwise Image Similarity Mining"
echo "Mode:        ${MODE}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Date:        $(date)"
echo "Job Dir:     ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP
# ------------------------------------------------------------------
export WANDB_DIR="$LOGS_DIR"

if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
    echo "W&B API Key not found. Running offline."
else
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_pairwise_${MODE}"
fi

# ------------------------------------------------------------------
# OUTPUT PATH
# ------------------------------------------------------------------
OUTPUT_PATH="datasets/coco/pairwise_mining_${MODE}.pt"
echo "Output Path: ${OUTPUT_PATH}"

# ------------------------------------------------------------------
# RUN PAIRWISE MINING
# ------------------------------------------------------------------
cd /workspace

echo ""
echo "Starting pairwise mining process (mode=${MODE})..."
echo ""

python3 -u tools/calculate_pairwise_similarity.py \
    --config config.yaml \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --output "$OUTPUT_PATH" \
    2>&1 | tee "$LOGS_DIR/pairwise_mining_${MODE}.log"

MINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $MINING_EXIT_CODE -eq 0 ]; then
    echo "Pairwise mining (mode=${MODE}) completed successfully!"
    echo "Output: ${OUTPUT_PATH}"
    if [ -f "$OUTPUT_PATH" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "File size: ${FILE_SIZE}"
    fi
else
    echo "Pairwise mining failed with exit code: $MINING_EXIT_CODE"
fi
echo "=========================================="

exit $MINING_EXIT_CODE
