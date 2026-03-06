#!/bin/bash
set -e

echo "=========================================="
echo "  ECCV Benchmark — Batch Submission"
echo "=========================================="

CONFIG_FILE="${1:-configs/config_coco.yaml}"
SPLIT="${2:-test}"
MODE="${3:-normal}" # 'normal' veya 'eccv'

# DİKKAT: Bu yollar konteynerin İÇİNDEKİ (mounted) yollardır.
MODELS=(
    "/output/results/coco/models/coco_baseline_256.pth"
    "/output/results/coco/models/coco_fne_0.99.pth"
    "/output/results/coco/models/mixed_img_aug.pth"
    "openai/clip-vit-large-patch14-336"
)

for CKPT in "${MODELS[@]}"; do
    # Model ismini yoldan çıkar (Loglama için)
    MODEL_NAME=$(basename "$CKPT" .pth)
    JOB_NAME="eval_${MODEL_NAME}"
    
    echo "Submitting: $JOB_NAME"
    echo "  Checkpoint: $CKPT"
    
    sbatch --job-name="$JOB_NAME" scripts/eval.slurm "$CONFIG_FILE" "$CKPT" "$SPLIT" "$MODE"
done

echo "All jobs submitted."
