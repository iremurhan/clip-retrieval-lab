#!/bin/bash
# =============================================================================
# SLURM EVALUATION SCRIPT — Standard + ECCV Modes
# =============================================================================
# Evaluates a DualEncoder model on COCO/Flickr30k (standard R@K/MAP) or
# runs the ECCV Caption benchmark protocol (eccv_caption package).
#
# Dispatched by:  scripts/start_eval.sh (recommended)
# Direct usage:   sbatch scripts/eval.slurm <CONFIG> <CHECKPOINT> <SPLIT> <MODE>
#
# Arguments (positional):
#   $1  CONFIG       Config file path          (default: configs/config_coco.yaml)
#   $2  CHECKPOINT   .pth path or HF model ID  (required)
#   $3  SPLIT        val | test                (default: test)
#   $4  MODE         normal | eccv             (default: normal)
#
# Examples:
#   sbatch scripts/eval.slurm configs/config_coco.yaml /output/results/coco/models/best.pth test normal
#   sbatch scripts/eval.slurm configs/config_coco.yaml /output/results/coco/models/best.pth test eccv
#   sbatch scripts/eval.slurm configs/config_coco.yaml openai/clip-vit-large-patch14-336 test normal
#
# Output: /output/results/eval_{SLURM_JOB_ID}/  (eval.log, eval_slurm.out)
# NOTE:   ECCV mode is only available for COCO. Flickr must use MODE=normal.
# =============================================================================
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/eval_slurm.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/eval_slurm.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

CONFIG_FILE="${1:-configs/config_coco.yaml}"
CHECKPOINT="${2}"
SPLIT="${3:-test}"
MODE="${4:-normal}"

# Set output directory
JOB_DIR="/output/results/coco/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Mode:          ${MODE}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output Dir:    ${JOB_DIR}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

# Install eccv_caption library only if ECCV mode is selected
if [ "$MODE" == "eccv" ]; then
    echo "ECCV Mode detected. Installing eccv_caption library..."
    pip install eccv_caption
    ECCV_FLAG="--eccv"
else
    ECCV_FLAG=""
fi

echo "Running Evaluation..."

python3 -u tools/eval.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    $ECCV_FLAG \
    > "$JOB_DIR/eval.log" 2>&1

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi

exit $EVAL_EXIT_CODE
