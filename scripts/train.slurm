#!/bin/bash
# Training. Prefer: ./scripts/start_training.sh <run_name> [config]
# Direct: sbatch --job-name=NAME scripts/train.slurm NAME [config]
# Logs: results/{coco|flickr30k}/{job_id}/ (slurm.out, training.log, checkpoints/, wandb/). W&B: retrieval.
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/{DATASET}/%j/slurm.out
#SBATCH --error=/users/beyza.urhan/experiments/results/{DATASET}/%j/slurm.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

EXP_NAME="${1:-}"
CONFIG_FILE="${2:-configs/config_coco.yaml}"

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi
WANDB_PROJECT_AUTO="retrieval"

JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
CKPT_DIR="$JOB_DIR/checkpoints"
mkdir -p "$JOB_DIR" "$CKPT_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Experiment:    ${EXP_NAME:- (Unnamed)}"
echo "Dataset:       ${DATASET}"
echo "Job dir:       ${JOB_DIR} (slurm.out, training.log, checkpoints/, wandb/)"
echo "=========================================="

export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    if [ -n "$EXP_NAME" ]; then
        export WANDB_NAME="${SLURM_JOB_ID}_${EXP_NAME}"
    else
        export WANDB_NAME="${SLURM_JOB_ID}"
    fi

    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    echo "W&B Project auto-set: $WANDB_PROJECT"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

if [ -f "/workspace/$CONFIG_FILE" ]; then
    cp "/workspace/$CONFIG_FILE" "$JOB_DIR/config.yaml"
fi

echo "Starting training..."
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

python3 -u run.py \
    --config "$CONFIG_FILE" \
    --override \
        debug.debug_mode=false \
        logging.checkpoint_dir="$CKPT_DIR" \
        logging.wandb_project="$WANDB_PROJECT_AUTO" \
    > "$JOB_DIR/training.log" 2>&1

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Results saved to: $JOB_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
