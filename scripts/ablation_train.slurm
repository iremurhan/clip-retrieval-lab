#!/bin/bash
# =============================================================================
# SLURM TRAINING SCRIPT — Ablation launcher variant
# =============================================================================
# Accepts run_id, seed, and config as positional arguments.
# Called by scripts/launch_ablation.sh — do not invoke directly.
#
# Usage (via sbatch):
#   sbatch scripts/ablation_train.slurm <RUN_ID> <SEED> <CONFIG>
# Example:
#   sbatch scripts/ablation_train.slurm B0plus 42 configs/config_flickr30k.yaml
#
# Arguments:
#   $1  RUN_ID   Registry run ID (e.g. B0, FULL)
#   $2  SEED     Random seed (e.g. 42)
#   $3  CONFIG   Dataset config path (e.g. configs/config_coco.yaml)
#
# Checkpoint is written to:
#   /output/results/{dataset}/{SLURM_JOB_ID}/  (best_model.pth, last_model.pth)
# =============================================================================
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

RUN_ID="${1}"
SEED="${2}"
CONFIG_FILE="${3:-configs/config_coco.yaml}"

DATASET=$(python3 -c "
import sys, yaml
base = yaml.safe_load(open('configs/config_base.yaml'))
override = yaml.safe_load(open('${CONFIG_FILE}'))
base.setdefault('data', {}).update(override.get('data', {}))
d = base['data'].get('dataset')
if not d:
    sys.exit('ERROR: data.dataset not set in config')
print(d)
") || exit 1

WANDB_PROJECT_AUTO="clip-retrieval"
JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
CKPT_DIR="$JOB_DIR"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Run ID:     ${RUN_ID}"
echo "Seed:       ${SEED}"
echo "Dataset:    ${DATASET}"
echo "Config:     ${CONFIG_FILE}"
echo "Job dir:    ${JOB_DIR}"
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

echo "Starting training..."
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
python3 -u run.py \
    --config "$CONFIG_FILE" \
    --run "$RUN_ID" \
    --seed "$SEED" \
    --override \
        logging.checkpoint_dir="$CKPT_DIR" \
    > "$JOB_DIR/training.log" 2>&1
TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Results saved to: $JOB_DIR"
echo "=========================================="

# Attempt to sync WandB logs before any cleanup.
echo "Syncing WandB logs before cleanup..."
SYNC_EXIT=1
if [ -d "$JOB_DIR/wandb" ]; then
    python3 -m wandb sync --sync-all "$JOB_DIR/wandb" > "$JOB_DIR/wandb_sync.log" 2>&1
    SYNC_EXIT=$?
fi

if [ $SYNC_EXIT -eq 0 ]; then
    echo "WandB sync successful."
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        if [ -f "$JOB_DIR/last_model.pth" ]; then
            rm -f "$JOB_DIR/last_model.pth"
        fi
    fi
    if [ -d "$JOB_DIR/wandb" ]; then
        rm -rf "$JOB_DIR/wandb"
    fi
else
    echo "WARNING: WandB sync failed. Keeping '$JOB_DIR/wandb' and 'last_model.pth' for manual recovery."
fi

# Move slurm log into the dataset/job directory now that JOB_DIR is known.
SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $TRAIN_EXIT_CODE
