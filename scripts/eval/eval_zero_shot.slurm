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
#   sbatch scripts/eval/eval_zero_shot.slurm <dataset> <input_mode> [seed] [wandb_run_id]
# Examples:
#   sbatch scripts/eval/eval_zero_shot.slurm flickr30k rgb
#   sbatch scripts/eval/eval_zero_shot.slurm flickr30k mask
#   sbatch scripts/eval/eval_zero_shot.slurm coco      rgb
#   sbatch scripts/eval/eval_zero_shot.slurm coco      mask
#
# Outputs:
#   /output/results/zero_shot/<jobid>/<dataset>_<mode>.json      (metrics)
#   /output/results/zero_shot/<jobid>/<dataset>_<mode>.log       (stdout/err)
#
# No checkpoint, no training side-effects. Retrieval, SugarCrepe, and MMVP-VLM
# metrics are logged to W&B when /env/wandb.env provides WANDB_API_KEY.
# =============================================================================
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/zero_shot_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/zero_shot_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch scripts/eval/eval_zero_shot.slurm <coco|flickr30k> <rgb|mask>}"
INPUT_MODE="${2:?Usage: sbatch scripts/eval/eval_zero_shot.slurm <coco|flickr30k> <rgb|mask>}"
SEED="${3:-42}"
WANDB_RUN_ID="${4:-}"

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

OUT_DIR="/output/results/zero_shot/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
OUT_JSON="${OUT_DIR}/${DATASET}_${INPUT_MODE}.json"
OUT_LOG="${OUT_DIR}/${DATASET}_${INPUT_MODE}.log"
WANDB_PROJECT_AUTO="clip-retrieval"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET}"
echo "Input mode:  ${INPUT_MODE}"
echo "Seed:        ${SEED} (bookkeeping only; zero-shot test eval is deterministic)"
echo "W&B run ID:  ${WANDB_RUN_ID:-(new run)}"
echo "Config:      ${CONFIG}"
echo "Output JSON: ${OUT_JSON}"
echo "=========================================="

export WANDB_DIR="$OUT_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    echo "W&B Project auto-set: $WANDB_PROJECT"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

set +e
EVAL_CMD=(
    python3 -u tools/eval_zero_shot.py
    --config "$CONFIG"
    --input "$INPUT_MODE"
    --output "$OUT_JSON"
    --seed "$SEED"
    --log-wandb
    --wandb_project "$WANDB_PROJECT_AUTO"
)
if [ -n "$WANDB_RUN_ID" ]; then
    EVAL_CMD+=(--wandb_run_id "$WANDB_RUN_ID")
fi
EVAL_CMD+=(--wandb_run_name "zero_shot_clip_${DATASET}_${INPUT_MODE}")
"${EVAL_CMD[@]}" 2>&1 | tee "$OUT_LOG"
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

source /workspace/scripts/util/wandb_cleanup.sh
wandb_sync_and_cleanup "$OUT_DIR" "$WANDB_PROJECT_AUTO" "$EVAL_EXIT"

# Move the top-level slurm_<jobid>.out into the per-run log dir for tidiness.
SLURM_LOG="/users/beyza.urhan/experiments/results/zero_shot_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "${OUT_DIR}/slurm.out"
fi

exit $EVAL_EXIT
