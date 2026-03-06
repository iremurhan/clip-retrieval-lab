#!/bin/bash
# =============================================================================
# ECCV BENCHMARK — BATCH TRIGGER
# =============================================================================
# Submits one sbatch job per model checkpoint.
#
# Usage:
#   bash scripts/start_eval.sh
#
# Models:
#   1. coco_baseline_256      (fine-tuned)
#   2. coco_fne_0.99          (FNE custom)
#   3. mixed_img_aug          (augmentation variant)
#   4. clip-vit-large-patch14 (zero-shot, HuggingFace)
# =============================================================================

set -euo pipefail

CONFIG="configs/config_coco.yaml"

MODELS=(
    "/home/beyza.urhan/experiments/results/coco/models/coco_baseline_256.pth"
    "/home/beyza.urhan/experiments/results/coco/models/coco_fne_0.99.pth"
    "/home/beyza.urhan/experiments/results/coco/models/mixed_img_aug.pth"
    "openai/clip-vit-large-patch14-336"
)

echo "=========================================="
echo "  ECCV Benchmark — Batch Submission"
echo "=========================================="
echo "  Config: $CONFIG"
echo "  Models: ${#MODELS[@]}"
echo "=========================================="
echo ""

for CKPT in "${MODELS[@]}"; do
    # --- Derive job name from checkpoint basename ---
    if [[ "$CKPT" == *.pth ]]; then
        JOB_NAME=$(basename "$CKPT" .pth)
    else
        # HuggingFace model: use last component
        JOB_NAME=$(basename "$CKPT")
    fi

    # --- Pre-flight: verify .pth checkpoints exist ---
    if [[ "$CKPT" == *.pth && ! -f "$CKPT" ]]; then
        echo "SKIP: Checkpoint not found: $CKPT"
        continue
    fi

    echo "Submitting: eval_${JOB_NAME}"
    echo "  Checkpoint: $CKPT"

    sbatch \
        --job-name="eval_${JOB_NAME}" \
        --export=ALL,EVAL_CONFIG="$CONFIG",EVAL_CHECKPOINT="$CKPT" \
        scripts/eval.slurm

    echo ""
done

echo "All jobs submitted."
