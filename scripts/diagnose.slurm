#!/bin/bash
#
# Usage:
#   sbatch --job-name="DIAG_FLICKR" scripts/diagnose.slurm flickr
#   sbatch --job-name="DIAG_COCO"   scripts/diagnose.slurm coco
#
# This script mirrors the training SLURM setup but runs failure analysis
# with tools/diagnose_model.py instead of train.py.
#

#SBATCH --output=/users/beyza.urhan/experiments/runs/%j/diagnose.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j/diagnose_err.log
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr

# ------------------------------------------------------------------
# PATH MAPPING
# ------------------------------------------------------------------
# NOTE: Adjust these host paths if your user / dataset layout differs.
# Here we assume:
#   - Code:      /users/beyza.urhan/tez_v2_clean           -> /workspace
#   - COCO:      /users/beyza.urhan/experiments/datasets/coco -> /workspace/datasets/coco
#   - Flickr30k: /users/beyza.urhan/experiments/datasets/flickr30k -> /workspace/datasets/flickr30k
#   - Runs:      /users/beyza.urhan/experiments/runs       -> /output
#   - Env:       /users/beyza.urhan/experiments/env        -> /env
#
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env

# Pass W&B Env vars
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
DATASET="${1:-flickr}"   # "flickr" or "coco"

JOB_DIR="/output/${SLURM_JOB_ID}"
LOGS_DIR="$JOB_DIR/logs"
META_DIR="$JOB_DIR/metadata"
OUT_DIR="$JOB_DIR/diagnosis"

mkdir -p "$JOB_DIR" "$LOGS_DIR" "$META_DIR" "$OUT_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Dataset:       ${DATASET}"
echo "Job Directory: ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP (optional, for consistency with training)
# ------------------------------------------------------------------
export WANDB_DIR="$LOGS_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

# ------------------------------------------------------------------
# METADATA
# ------------------------------------------------------------------
echo "Job ID: ${SLURM_JOB_ID}" > "$META_DIR/diagnose_metadata.txt"
echo "Dataset: ${DATASET}" >> "$META_DIR/diagnose_metadata.txt"
echo "Date: $(date)" >> "$META_DIR/diagnose_metadata.txt"

cd /workspace

echo "Starting failure analysis..."

# ------------------------------------------------------------------
# DATASET-SPECIFIC ARGS
# ------------------------------------------------------------------

if [ "$DATASET" = "flickr" ]; then
    CONFIG="config_flickr.yaml"
    CHECKPOINT="/output/flickr_best_model.pth"   # adjust if needed
    IMAGE_ROOT="/workspace/datasets/flickr30k/flickr30k_images"
    CAPTIONS_PATH="/workspace/datasets/flickr30k/caption_datasets/dataset_flickr30k.json"
    DIRECTION="both"
elif [ "$DATASET" = "coco" ]; then
    CONFIG="config_coco.yaml"
    CHECKPOINT="/output/coco_best_model.pth"     # adjust if needed
    IMAGE_ROOT="/workspace/datasets/coco"
    CAPTIONS_PATH="/workspace/datasets/coco/caption_datasets/dataset_coco.json"
    DIRECTION="both"
else
    echo "Unknown DATASET argument: $DATASET (expected 'flickr' or 'coco')"
    exit 1
fi

echo "Using:"
echo "  CONFIG        = $CONFIG"
echo "  CHECKPOINT    = $CHECKPOINT"
echo "  IMAGE_ROOT    = $IMAGE_ROOT"
echo "  CAPTIONS_PATH = $CAPTIONS_PATH"
echo "  OUTPUT_DIR    = $OUT_DIR"
echo "  DIRECTION     = $DIRECTION"

# ------------------------------------------------------------------
# RUN DIAGNOSE SCRIPT
# ------------------------------------------------------------------

python3 -u tools/diagnose_model.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --image_root "$IMAGE_ROOT" \
  --captions_path "$CAPTIONS_PATH" \
  --output_dir "$OUT_DIR" \
  --direction "$DIRECTION" \
  --max_failures 200 \
  2>&1 | tee "$LOGS_DIR/diagnose_output.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Failure analysis completed successfully!"
else
    echo "Failure analysis failed with exit code: $EXIT_CODE"
fi
echo "Results (JSON + HTML) in: $OUT_DIR"
echo "=========================================="

exit $EXIT_CODE

