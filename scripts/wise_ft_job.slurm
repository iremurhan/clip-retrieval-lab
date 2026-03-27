#!/bin/bash
# =============================================================================
# SLURM WiSE-FT JOB
# =============================================================================
# Dependency job submitted by launch_run.sh after a training job completes.
# Runs scripts/wise_ft.py on the best checkpoint produced by the training job.
#
# Usage (launched via launch_run.sh; can also be submitted manually):
#   sbatch --dependency=afterok:<TRAIN_JOB_ID> scripts/wise_ft_job.slurm \
#       --checkpoint /path/to/job_output_dir \
#       --config configs/config_coco.yaml
# =============================================================================
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.14
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/wise_ft.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/wise_ft.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

# Parse --checkpoint and --config from positional or named args
CHECKPOINT_DIR=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT_DIR="$2"; shift 2 ;;
        --config)     CONFIG_FILE="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT_DIR" ] || [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: --checkpoint and --config are required."
    exit 1
fi

BEST_CKPT="${CHECKPOINT_DIR}/best_model.pth"
if [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: best_model.pth not found in ${CHECKPOINT_DIR}"
    exit 1
fi

OUTPUT_DIR="/output/results/${SLURM_JOB_ID}/wise_ft"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Checkpoint:     ${BEST_CKPT}"
echo "Config:         ${CONFIG_FILE}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "=========================================="

if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

python3 -u scripts/wise_ft.py \
    --pretrained openai/clip-vit-large-patch14-336 \
    --finetuned  "$BEST_CKPT" \
    --config     "/workspace/${CONFIG_FILE}" \
    --alphas 0.3 0.5 0.7 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/wise_ft.log"

echo "WiSE-FT finished. Results in ${OUTPUT_DIR}"
