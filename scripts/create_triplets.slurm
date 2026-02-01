#!/bin/bash
#-------------------------------------------------------------------------------------------------
# Usage:
#   COCO:   sbatch --export=TARGET_DATASET=coco scripts/create_triplets.slurm
#   Flickr: sbatch --export=ALL,TARGET_DATASET=flickr30k scripts/create_triplets.slurm
# Prerequisite: run mining first so .pt files exist under datasets/{coco|flickr30k}/pairwise_similarities/
#-------------------------------------------------------------------------------------------------
#SBATCH --job-name=triplets
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/triplets_log.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/triplets_log.out
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# Mount settings
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env

# Pass WandB variables (Optional for this script but good practice)
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,TARGET_DATASET

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

# Define paths relative to /workspace mount
INPUT_JSON="datasets/${DATASET_NAME}/caption_datasets/dataset_${DATASET_NAME}.json"
MINING_DIR="datasets/${DATASET_NAME}/pairwise_similarities"
INDICES_PATH="${MINING_DIR}/mining_text_top1000_indices.pt"
VALUES_PATH="${MINING_DIR}/mining_text_top1000_values.pt"
OUTPUT_JSON="datasets/${DATASET_NAME}/triplets.json"

# Log handling paths
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
    echo "Job finished. Moving job directory to ${DATASET_NAME}..."
    sync
    if [ ! -d "$TARGET_PARENT_DIR" ]; then
        mkdir -p "$TARGET_PARENT_DIR"
    fi
    if [ -d "$CURRENT_JOB_DIR" ]; then
        mv "$CURRENT_JOB_DIR" "$TARGET_PARENT_DIR/"
        echo "Successfully moved to: ${TARGET_JOB_DIR}"
    fi
}
trap move_job_folder EXIT

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

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