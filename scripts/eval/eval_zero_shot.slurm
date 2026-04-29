#!/bin/bash
# =============================================================================
# Zero-shot CLIP retrieval evaluation (B5_mask_only ablation).
#
# Runs tools/eval_zero_shot.py with no training: pretrained CLIP is scored on
# the Karpathy test split under one of two input regimes:
#   rgb   -> original RGB images (the zero-shot baseline)
#   mask  -> colorized SAM segment masks via SegToRGBRenderer (structure-only)
#
# Usage:
#   sbatch scripts/eval/eval_zero_shot.slurm <dataset> <input_mode>
# Examples:
#   sbatch scripts/eval/eval_zero_shot.slurm flickr30k rgb
#   sbatch scripts/eval/eval_zero_shot.slurm flickr30k mask
#   sbatch scripts/eval/eval_zero_shot.slurm coco      rgb
#   sbatch scripts/eval/eval_zero_shot.slurm coco      mask
#
# Outputs:
#   /output/results/zero_shot/<dataset>_<mode>_<jobid>.json      (metrics)
#   /output/results/zero_shot/<dataset>_<mode>_<jobid>.log       (stdout/err)
#
# No WandB, no checkpoint, no training side-effects.
# =============================================================================
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:b5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results
#SBATCH --output=/users/beyza.urhan/experiments/results/zero_shot_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/zero_shot_%j.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch scripts/eval/eval_zero_shot.slurm <coco|flickr30k> <rgb|mask>}"
INPUT_MODE="${2:?Usage: sbatch scripts/eval/eval_zero_shot.slurm <coco|flickr30k> <rgb|mask>}"

case "$DATASET" in
    coco)       CONFIG="configs/config_coco.yaml" ;;
    flickr30k)  CONFIG="configs/config_flickr30k.yaml" ;;
    *) echo "Unknown dataset: $DATASET (expected coco or flickr30k)"; exit 1 ;;
esac

case "$INPUT_MODE" in
    rgb|mask) ;;
    *) echo "Unknown input mode: $INPUT_MODE (expected rgb or mask)"; exit 1 ;;
esac

if [ "$INPUT_MODE" = "mask" ]; then
    # Fail fast if SAM masks are not on disk for the chosen dataset.
    SEG_DIR="/workspace/datasets/${DATASET}/sam_masks"
    if [ ! -d "$SEG_DIR" ] || [ -z "$(ls -A "$SEG_DIR" 2>/dev/null)" ]; then
        echo "FATAL: mask mode requires precomputed SAM masks at $SEG_DIR (empty or missing)."
        echo "Run scripts/preprocess/precompute_sam.slurm for $DATASET first."
        exit 1
    fi
fi

OUT_DIR="/output/results/zero_shot"
mkdir -p "$OUT_DIR"
OUT_JSON="${OUT_DIR}/${DATASET}_${INPUT_MODE}_${SLURM_JOB_ID}.json"
OUT_LOG="${OUT_DIR}/${DATASET}_${INPUT_MODE}_${SLURM_JOB_ID}.log"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET}"
echo "Input mode:  ${INPUT_MODE}"
echo "Config:      ${CONFIG}"
echo "Output JSON: ${OUT_JSON}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

set +e
python3 -u tools/eval_zero_shot.py \
    --config "$CONFIG" \
    --input "$INPUT_MODE" \
    --output "$OUT_JSON" \
    2>&1 | tee "$OUT_LOG"
EVAL_EXIT=${PIPESTATUS[0]}
set -e

echo ""
echo "=========================================="
if [ $EVAL_EXIT -eq 0 ]; then
    echo "Zero-shot eval completed. Metrics: $OUT_JSON"
else
    echo "Zero-shot eval failed with exit code: $EVAL_EXIT"
fi
echo "=========================================="

# Move the top-level slurm_<jobid>.out into the per-run log dir for tidiness.
SLURM_LOG="/users/beyza.urhan/experiments/results/zero_shot_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "${OUT_DIR}/${DATASET}_${INPUT_MODE}_${SLURM_JOB_ID}.slurm.out"
fi

exit $EVAL_EXIT
