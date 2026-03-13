#!/bin/bash
#SBATCH --job-name=triplets
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/triplets_log.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/triplets_log.out
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
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

if [ -d "/workspace" ]; then
    # Container Environment
    WORKSPACE_ROOT="/workspace"
    INPUT_JSON="/workspace/datasets/${DATASET_NAME}/caption_datasets/dataset_${DATASET_NAME}.json"
    MINING_DIR="/workspace/datasets/${DATASET_NAME}/pairwise_similarities"
    OUTPUT_JSON="/workspace/datasets/${DATASET_NAME}/triplets.json"
else
    # Host Environment (Fallback)
    echo ">>> WARNING: Running on HOST (No Container detected)"
    WORKSPACE_ROOT="${SLURM_SUBMIT_DIR:-.}"
    # Symlink sayesinde datasets klasörüne erişiyoruz
    INPUT_JSON="${WORKSPACE_ROOT}/datasets/${DATASET_NAME}/caption_datasets/dataset_${DATASET_NAME}.json"
    MINING_DIR="${WORKSPACE_ROOT}/datasets/${DATASET_NAME}/pairwise_similarities"
    OUTPUT_JSON="${WORKSPACE_ROOT}/datasets/${DATASET_NAME}/triplets.json"
fi

INDICES_PATH="${MINING_DIR}/mining_text_top1000_indices.pt"
VALUES_PATH="${MINING_DIR}/mining_text_top1000_values.pt"

CURRENT_JOB_DIR="/output/results/${SLURM_JOB_ID}"
TARGET_PARENT_DIR="/output/results/${DATASET_NAME}"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET_NAME}"
echo "Workspace:   ${WORKSPACE_ROOT}"
echo "=========================================="

# Trap Function
move_job_folder() {
    # Sadece /output varsa (yani container içindeysek) log taşıma yap
    if [ -d "/output" ]; then
        echo "Job finished. Moving logs..."
        sync
        if [ -d "$CURRENT_JOB_DIR" ]; then
            mkdir -p "$TARGET_PARENT_DIR"
            mv "$CURRENT_JOB_DIR" "$TARGET_PARENT_DIR/" || echo "Log taşıma uyarısı."
        fi
    fi
}
trap move_job_folder EXIT

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd "$WORKSPACE_ROOT"

# Dosya Kontrolü
if [ ! -f "$INDICES_PATH" ]; then
    echo "ERROR: Mining indices not found at $INDICES_PATH"
    echo "Please run mining first!"
    exit 1
fi

python3 tools/create_triplets.py \
    --dataset_json "$INPUT_JSON" \
    --mining_indices "$INDICES_PATH" \
    --mining_values "$VALUES_PATH" \
    --output_path "$OUTPUT_JSON" \
    --num_negatives 1 \
    --threshold 0.90 \
    --split train

echo "Done! Output: $OUTPUT_JSON"