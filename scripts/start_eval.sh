#!/bin/bash
# =============================================================================
# BATCH EVALUATION TRIGGER
# =============================================================================
# Submits one sbatch job per model checkpoint via scripts/eval.slurm.
# Each job evaluates a DualEncoder model in either standard or ECCV mode.
#
# Usage:
#   bash scripts/start_eval.sh [CONFIG] [SPLIT] [MODE]
#
# Arguments (positional, all optional):
#   CONFIG   Config file path   (default: configs/config_coco.yaml)
#   SPLIT    val | test         (default: test)
#   MODE     normal | eccv      (default: normal)
#
# Examples:
#   bash scripts/start_eval.sh                                          # defaults (COCO, test, normal)
#   bash scripts/start_eval.sh configs/config_coco.yaml test eccv       # ECCV mode (COCO only)
#   bash scripts/start_eval.sh configs/config_flickr.yaml val normal    # Flickr val (standard only)
#
# NOTE: ECCV mode is only available for COCO. Flickr must use 'normal' mode.
#
# Models submitted (4 jobs):
#   1. coco_baseline_256.pth    (fine-tuned baseline)
#   2. coco_fne_0.99.pth        (FNE custom)
#   3. mixed_img_aug.pth        (augmentation variant)
#   4. openai/clip-vit-large    (zero-shot, HuggingFace)
# =============================================================================
set -e

echo "=========================================="
echo "  ECCV Benchmark — Batch Submission"
echo "=========================================="

CONFIG_FILE="${1:-configs/config_coco.yaml}"
SPLIT="${2:-test}"
MODE="${3:-normal}" # 'normal' or 'eccv'

# NOTE: These paths are INSIDE the container (mounted paths).
MODELS=(
    "/output/results/coco/models/2-coco_baseline_256.pth"
    "/output/results/coco/models/2-coco_fne_0.99.pth"
    "/output/results/coco/models/2-mixed_img_aug.pth"
    "openai/clip-vit-large-patch14-336"
)

for CKPT in "${MODELS[@]}"; do
    # Extract model name from path (for logging)
    MODEL_NAME=$(basename "$CKPT" .pth)
    JOB_NAME="eval_${MODEL_NAME}"
    
    echo "Submitting: $JOB_NAME"
    echo "  Checkpoint: $CKPT"
    
    sbatch --job-name="$JOB_NAME" scripts/eval.slurm "$CONFIG_FILE" "$CKPT" "$SPLIT" "$MODE"
done

echo "All jobs submitted."
