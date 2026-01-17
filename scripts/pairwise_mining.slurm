#!/bin/bash
#
# Pairwise Image Similarity Mining (End-to-End Joint MAX/MIN)
# Usage: sbatch scripts/pairwise_mining.slurm
#
# This script performs end-to-end mining:
# 1. Loads CLIP model and extracts caption embeddings
# 2. Performs joint MAX/MIN pairwise similarity mining
# 3. Saves results
#
#SBATCH --job-name=pairwise_mining
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j/slurm.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j/err.log
#SBATCH --time=02-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,QUERY_CHUNK_SIZE,TOP_K,EXTRACTION_BATCH_SIZE

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
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-100}"
TOP_K="${TOP_K:-50}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-512}"

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
    export WANDB_NAME="${SLURM_JOB_ID}_pairwise_joint_maxmin"
fi

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

OUTPUT_PATH="datasets/coco/pairwise_mining_joint.pt"

echo "=========================================="
echo "Job ID:                ${SLURM_JOB_ID}"
echo "Task:                  Pairwise Mining (End-to-End)"
echo "Mode:                  joint_maxmin"
echo "Output Path:           ${OUTPUT_PATH}"
echo "Top-K:                 ${TOP_K}"
echo "Query Chunk Size:      ${QUERY_CHUNK_SIZE}"
echo "Extraction Batch Size: ${EXTRACTION_BATCH_SIZE}"
echo "Job Dir:               ${JOB_DIR}"
echo "=========================================="

echo "Starting end-to-end mining process..."
echo "  Phase 1: Load CLIP Model & Dataset"
echo "  Phase 2: Extract Caption Embeddings"
echo "  Phase 3: Free CLIP Model"
echo "  Phase 4: Joint MAX/MIN Mining"
echo "  Phase 5: Save Results"

# Run Python script
# Redirect stdout/stderr to both console (Slurm log) and a specific file in Job Dir
python3 -u tools/calculate_pairwise_similarity.py \
    --config config.yaml \
    --output_path "$OUTPUT_PATH" \
    --top_k "$TOP_K" \
    --query_chunk_size "$QUERY_CHUNK_SIZE" \
    --extraction_batch_size "$EXTRACTION_BATCH_SIZE" \
    2>&1 | tee "$JOB_DIR/mining_process.log"