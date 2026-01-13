#!/bin/bash
#
# Pairwise Image Similarity Mining
# Usage: sbatch scripts/pairwise_mining.slurm [--export=MODE=mean]
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

# ------------------------------------------------------------------
# PATH MAPPING (Identical to train.slurm)
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env

# W&B Environment Variables
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

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
MODE="${MODE:-mean}"
BATCH_SIZE="${BATCH_SIZE:-512}"

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
OUTPUT_PATH="datasets/coco/pairwise_mining_${MODE}.pt"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Task:        Pairwise Mining"
echo "Mode:        ${MODE}"
echo "Output:      ${OUTPUT_PATH}"
echo "Job Dir:     ${JOB_DIR}"
echo "=========================================="

echo "Starting mining process..."

# Run Python script
# Redirect stdout/stderr to both console (Slurm log) and a specific file in Job Dir
python3 -u tools/calculate_pairwise_similarity.py \
    --config config.yaml \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --output "$OUTPUT_PATH" \
    2>&1 | tee "$JOB_DIR/mining_process.log"