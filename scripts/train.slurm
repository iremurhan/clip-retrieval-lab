#!/bin/bash
#
# Usage:
#   sbatch --job-name="EXP_NAME" scripts/train.slurm "EXP_NAME" ["CONFIG_PATH"]
#
# Output Structure:
#   experiments/results/{dataset}/{job_id}/
#     - slurm.out      (script/SLURM: container, setup, job messages)
#     - training.log   (Python/training output only)
#     - checkpoints/   (model checkpoints)
#     - wandb/         (WandB files if enabled)
#
# The W&B project is set automatically:
#   - If the config path contains "flickr":  retrieval-flickr30k
#   - Otherwise:                             retrieval-coco
#
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr

# Mounts: if NOT MOUNTED on cluster, replace ${HOME} with your home path (echo $HOME)
#SBATCH --container-mounts=${HOME}/tez_v2_clean:/workspace,${HOME}/experiments/datasets/coco:/workspace/datasets/coco,${HOME}/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,${HOME}/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,${HOME}/experiments/results:/output/results,${HOME}/experiments/env:/env
#SBATCH --output=${HOME}/experiments/results/%j/slurm.out
#SBATCH --error=${HOME}/experiments/results/%j/slurm.out
# Pass W&B Env vars and PYTHONPATH so container finds src
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

# ------------------------------------------------------------------
# MOUNT VERIFICATION
# ------------------------------------------------------------------
echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
EXP_NAME="${1:-}"
CONFIG_FILE="${2:-configs/config_coco.yaml}"

# Validate config file (use explicit path, e.g. configs/config_coco.yaml)
if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Determine dataset from config file path
if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
    WANDB_PROJECT_AUTO="retrieval-flickr30k"
else
    DATASET="coco"
    WANDB_PROJECT_AUTO="retrieval-coco"
fi

# Job Directory Structure: results/{dataset}/{job_id}/
JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
CKPT_DIR="$JOB_DIR/checkpoints"

# Create directories
mkdir -p "$JOB_DIR" "$CKPT_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Experiment:    ${EXP_NAME:- (Unnamed)}"
echo "Dataset:       ${DATASET}"
echo "Job Directory: ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP
# ------------------------------------------------------------------
export WANDB_DIR="$JOB_DIR"

# Load W&B Env file if mounted
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

# WANDB_PROJECT_AUTO already set above based on dataset

# Configure W&B mode based on API key
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    
    # W&B Name Format: ID_NAME (e.g., 20362_vit_baseline)
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

# ------------------------------------------------------------------
# METADATA SAVING
# ------------------------------------------------------------------
if [ -f "/workspace/$CONFIG_FILE" ]; then
    cp "/workspace/$CONFIG_FILE" "$JOB_DIR/config.yaml"
fi

# ------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------
echo "Starting training..."

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

# Run training (Python output only -> training.log; script output -> slurm.out)
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
