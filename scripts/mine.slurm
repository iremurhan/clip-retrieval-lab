#!/bin/bash
#-------------------------------------------------------------------------------------------------
# Usage:
#   COCO: sbatch --export=TARGET_DATASET=coco scripts/mine.slurm or sbatch scripts/mine.slurm
#   Flickr30k: sbatch --export=TARGET_DATASET=flickr30k scripts/mine.slurm or sbatch --export=ALL,TARGET_DATASET=flickr30k scripts/mine.slurm
#
# With wrapper:
#   ./scripts/submit_mine.sh coco
#   ./scripts/submit_mine.sh flickr30k
#-------------------------------------------------------------------------------------------------
#SBATCH --job-name=mining
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/mining_log.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/mining_log.out
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# Mount settings
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env

# Pass WandB variables to container
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,TARGET_DATASET

set -euo pipefail

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Default to 'coco' if TARGET_DATASET is not set
DATASET="${TARGET_DATASET:-coco}"

if [ "$DATASET" == "flickr30k" ] || [ "$DATASET" == "flickr" ]; then
    DATASET_NAME="flickr30k"
    CONFIG_FILE="configs/config_flickr.yaml"
    echo ">>> MODE: FLICKR30K"
elif [ "$DATASET" == "coco" ]; then
    DATASET_NAME="coco"
    CONFIG_FILE="configs/config_coco.yaml"
    echo ">>> MODE: MS-COCO"
else
    echo "ERROR: Unknown dataset '$DATASET'. Use 'coco' or 'flickr30k'."
    exit 1
fi
CURRENT_JOB_DIR="/output/results/${SLURM_JOB_ID}"

TARGET_PARENT_DIR="/output/results/${DATASET_NAME}"
TARGET_JOB_DIR="${TARGET_PARENT_DIR}/${SLURM_JOB_ID}"


echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET_NAME}"
echo "Current:       ${CURRENT_JOB_DIR}"
echo "Target:       ${TARGET_JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# FINALIZE / MOVE LOGIC
# ------------------------------------------------------------------
# Function to move the job directory to the dataset folder upon exit
move_job_folder() {
    echo "Job finished. Moving job directory to ${DATASET_NAME}..."
    sync
    
    # Ensure target parent exists (safety check, even if it usually exists)
    if [ ! -d "$TARGET_PARENT_DIR" ]; then
        mkdir -p "$TARGET_PARENT_DIR"
    fi

    # Move the folder: results/JOB_ID -> results/DATASET_NAME/JOB_ID
    if [ -d "$CURRENT_JOB_DIR" ]; then
        mv "$CURRENT_JOB_DIR" "$TARGET_PARENT_DIR/"
        echo "Successfully moved to: ${TARGET_PARENT_DIR}/${SLURM_JOB_ID}"
    else
        echo "WARNING: Current job directory not found at $CURRENT_JOB_DIR"
    fi
}
# Execute move_job_folder when script exits (success or error)
trap move_job_folder EXIT

# ------------------------------------------------------------------
# WANDB SETUP
# ------------------------------------------------------------------
# Load wandb.env from /env if present
if [ -f /env/wandb.env ]; then
    echo "Loading WandB environment file..."
    set -a
    source /env/wandb.env
    set +a
else
    echo "WARNING: /env/wandb.env not found. API key may be missing."
fi

# Set project name here (overrides file). Set your project name in quotes:
export WANDB_PROJECT="mining"

# Set run name dynamically (include Job ID)
export WANDB_NAME="mining_${DATASET}_top1000_${SLURM_JOB_ID}"

echo "Active WandB project: $WANDB_PROJECT"

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

echo "Starting mining... for $DATASET"

python3 -u tools/mine_pairwise_sim.py \
    --modality caption \
    --config "$CONFIG_FILE" \
    --top_k 1000 \
    --batch_size 512 \
    --query_chunk_size 5000

echo "Mining completed... for $DATASET"
