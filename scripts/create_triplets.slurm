#!/bin/bash
#SBATCH --job-name=triplets
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/triplets_log.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/triplets_log.out
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,TARGET_DATASET
#-------------------------------------------------------------------------------------------------
# Usage (from repo root):
#   Container: sbatch --export=TARGET_DATASET=coco scripts/create_triplets.slurm
#   No container (data in ~/experiments/datasets):
#     sbatch --export=TARGET_DATASET=coco,DATASETS_ROOT=$HOME/experiments/datasets scripts/create_triplets.slurm
# Prerequisite: mining .pt files under {DATASETS_ROOT}/{coco|flickr30k}/pairwise_similarities/
#-------------------------------------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DATASET="${TARGET_DATASET:-coco}"

if [ "$DATASET" == "flickr30k" ] || [ "$DATASET" == "flickr" ]; then
    DATASET_NAME="flickr30k"
    echo ">>> MODE: FLICKR30K"
elif [ "$DATASET" == "coco" ]; then
    DATASET_NAME="coco"
    echo ">>> MODE: MS-COCO"
else
    echo "ERROR: Unknown dataset '$DATASET'. Use 'coco' or 'flickr'."
    exit 1
fi

# Project root: use /workspace if container mount exists, else submission dir
if [ -d "/workspace" ]; then
    WORKSPACE_ROOT="/workspace"
    DATASETS_ROOT="${DATASETS_ROOT:-$WORKSPACE_ROOT/datasets}"
else
    WORKSPACE_ROOT="${SLURM_SUBMIT_DIR:-.}"
    DATASETS_ROOT="${DATASETS_ROOT:-$HOME/experiments/datasets}"
    echo ">>> /workspace not found, using WORKSPACE_ROOT=$WORKSPACE_ROOT, DATASETS_ROOT=$DATASETS_ROOT"
fi
cd "$WORKSPACE_ROOT" || { echo "ERROR: Cannot cd to $WORKSPACE_ROOT"; exit 1; }

# Data paths (under DATASETS_ROOT so they work with or without container)
INPUT_JSON="${DATASETS_ROOT}/${DATASET_NAME}/caption_datasets/dataset_${DATASET_NAME}.json"
MINING_DIR="${DATASETS_ROOT}/${DATASET_NAME}/pairwise_similarities"
INDICES_PATH="${MINING_DIR}/mining_text_top1000_indices.pt"
VALUES_PATH="${MINING_DIR}/mining_text_top1000_values.pt"
OUTPUT_JSON="${DATASETS_ROOT}/${DATASET_NAME}/triplets.json"

# Log handling (only when running in container with /output mount)
CURRENT_JOB_DIR="/output/results/${SLURM_JOB_ID}"
TARGET_PARENT_DIR="/output/results/${DATASET_NAME}"
TARGET_JOB_DIR="${TARGET_PARENT_DIR}/${SLURM_JOB_ID}"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET_NAME}"
echo "Input JSON:  ${INPUT_JSON}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=========================================="

# ------------------------------------------------------------------
# FINALIZE / MOVE LOGIC (Logs will be moved to results/{dataset}/{job_id})
# ------------------------------------------------------------------
move_job_folder() {
    if [ ! -d "/output" ] || [ ! -w "/output" ]; then
        echo "Skipping log move (/output not available)."
        return 0
    fi
    echo "Job finished. Moving job directory to ${DATASET_NAME}..."
    sync
    if [ ! -d "$TARGET_PARENT_DIR" ]; then
        mkdir -p "$TARGET_PARENT_DIR" || { echo "WARNING: Could not mkdir $TARGET_PARENT_DIR"; return 0; }
    fi
    if [ -d "$CURRENT_JOB_DIR" ]; then
        mv "$CURRENT_JOB_DIR" "$TARGET_PARENT_DIR/" && echo "Moved to ${TARGET_PARENT_DIR}/${SLURM_JOB_ID}"
    fi
}
trap move_job_folder EXIT

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
echo "Starting triplet generation for $DATASET_NAME..."

# Check if mining inputs exist
if [ ! -f "$INDICES_PATH" ]; then
    echo "ERROR: Mining indices not found at $INDICES_PATH"
    echo "Please run mining first: ./scripts/start_mining.sh $DATASET_NAME"
    exit 1
fi
if [ ! -f "$VALUES_PATH" ]; then
    echo "ERROR: Mining values not found at $VALUES_PATH"
    echo "Please run mining first: ./scripts/start_mining.sh $DATASET_NAME"
    exit 1
fi
mkdir -p "$(dirname "$OUTPUT_JSON")"

python3 tools/create_triplets.py \
    --dataset_json "$INPUT_JSON" \
    --mining_indices "$INDICES_PATH" \
    --mining_values "$VALUES_PATH" \
    --output_path "$OUTPUT_JSON" \
    --num_negatives 1 \
    --threshold 0.90 \
    --split train

echo "Triplet generation completed. Output saved to: $OUTPUT_JSON"