#!/bin/bash
#
# Pairwise Image Similarity Mining (Fast Matrix Multiplication)
# Usage: sbatch scripts/pairwise_mining.slurm [MODE]
#        MODE: simple (default) or joint_maxmin
#
# Examples:
#   sbatch scripts/pairwise_mining.slurm simple
#   sbatch scripts/pairwise_mining.slurm joint_maxmin
#
#SBATCH --job-name=pairwise_mining
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j/slurm.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j/err.log
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,MODE,BATCH_SIZE,QUERY_CHUNK_SIZE,TOP_K

set -euo pipefail

# ------------------------------------------------------------------
# CRITICAL CHECK: MOUNT VERIFICATION
# ------------------------------------------------------------------
if [ ! -d "/output" ]; then
    echo "FATAL ERROR: '/output' directory does not exist inside the container."
    echo "This indicates a mount failure. Exiting to prevent writing to host root."
    exit 1
fi

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
# MODE from: 1) positional arg, 2) env var, 3) default
MODE="${1:-${MODE:-simple}}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-100}"
TOP_K="${TOP_K:-50}"

# Job Directory: /output/<JOB_ID> (Matches train.slurm logic)
JOB_DIR="/output/${SLURM_JOB_ID}"

# Create the directory immediately so Slurm log has a place to exist (if not already created by system)
mkdir -p "$JOB_DIR"

# W&B Setup
export WANDB_DIR="$JOB_DIR"
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
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

# Set paths based on mode
if [ "$MODE" = "joint_maxmin" ]; then
    FEATURES_PATH="experiments/features/coco_train_features.pt"
    OUTPUT_PATH="datasets/coco/pairwise_mining_joint.pt"
else
    # Simple mode - adjust paths as needed
    FEATURES_PATH="experiments/features/coco_train_features.pt"
    OUTPUT_PATH="datasets/coco/pairwise_mining_simple.pt"
fi

echo "=========================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Task:             Pairwise Mining (Fast)"
echo "Mode:             ${MODE}"
echo "Features Path:   ${FEATURES_PATH}"
echo "Output Path:      ${OUTPUT_PATH}"
echo "Top-K:            ${TOP_K}"
echo "Batch Size:       ${BATCH_SIZE}"
if [ "$MODE" = "joint_maxmin" ]; then
    echo "Query Chunk Size: ${QUERY_CHUNK_SIZE}"
fi
echo "Job Dir:          ${JOB_DIR}"
echo "=========================================="

echo "Starting mining process..."

# Run Python script
# Redirect stdout/stderr to both console (Slurm log) and a specific file in Job Dir
if [ "$MODE" = "joint_maxmin" ]; then
    python3 -u tools/calculate_pairwise_similarity.py \
        --mode "$MODE" \
        --config config.yaml \
        --features_path "$FEATURES_PATH" \
        --output_path "$OUTPUT_PATH" \
        --top_k "$TOP_K" \
        --query_chunk_size "$QUERY_CHUNK_SIZE" \
        2>&1 | tee "$JOB_DIR/mining_process.log"
else
    python3 -u tools/calculate_pairwise_similarity.py \
        --mode "$MODE" \
        --features_path "$FEATURES_PATH" \
        --output_path "$OUTPUT_PATH" \
        --top_k "$TOP_K" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee "$JOB_DIR/mining_process.log"
fi