#!/bin/bash
# One-off training launcher for result-only reruns: keep W&B metrics/logs,
# delete model checkpoints before sync/cleanup.
# Usage:
#   sbatch scripts/train/train_no_model_save.slurm <RUN_ID> <CONFIG_PATH> --seed <SEED>
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab/.codex_worktrees/consolidate-v1:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

RUN_ID="${1:?Usage: sbatch scripts/train/train_no_model_save.slurm <run_id> <config_path> [run.py args...]}"
CONFIG_FILE="${2:?Usage: sbatch scripts/train/train_no_model_save.slurm <run_id> <config_path> [run.py args...]}"
shift 2
EXTRA_ARGS=("$@")

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found."
    exit 1
fi

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

source /workspace/scripts/util/detect_dataset.sh
DATASET=$(detect_dataset "$CONFIG_FILE") || exit 1

WANDB_PROJECT_AUTO="clip-retrieval"
JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}_${RUN_ID}_result_only"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Run ID:     ${RUN_ID}"
echo "Dataset:    ${DATASET}"
echo "Config:     ${CONFIG_FILE}"
echo "Job dir:    ${JOB_DIR}"
echo "Extra args: ${EXTRA_ARGS[*]:-}"
echo "Mode:       result-only; checkpoints removed before W&B sync"
echo "=========================================="

export WANDB_DIR="$JOB_DIR"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
python3 -u run.py \
    --config "$CONFIG_FILE" \
    --run "$RUN_ID" \
    --override \
        logging.checkpoint_dir="$JOB_DIR" \
    "${EXTRA_ARGS[@]}" \
    > "$JOB_DIR/training.log" 2>&1
TRAIN_EXIT_CODE=$?
set -e

echo "Deleting model checkpoint files before W&B sync..."
find "$JOB_DIR" -maxdepth 1 -type f \( -name "*.pth" -o -name "*.pth.tmp" \) -delete

source /workspace/scripts/util/wandb_cleanup.sh
wandb_sync_and_cleanup "$JOB_DIR" "$WANDB_PROJECT_AUTO" "$TRAIN_EXIT_CODE"

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $TRAIN_EXIT_CODE
