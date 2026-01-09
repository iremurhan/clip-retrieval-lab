#!/bin/bash
#SBATCH -p normal
#SBATCH --time=230:59:59
#SBATCH --account=root
#SBATCH --job-name=irem
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/baytas/Documents/Irem/tez_v2_clean/scripts/Outputs/out-%j.out
#SBATCH --error=/home/baytas/Documents/Irem/tez_v2_clean/scripts/Errors/err-%j.err


module load Python/3.12.0 cuda/10.1

set -euo pipefail

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
EXP_NAME="${1:-}"


# 1. Job Directory (ALWAYS use Job ID only - Robust & Simple)
JOB_DIR=${SLURM_JOB_ID}

# Subdirectories
LOGS_DIR="$JOB_DIR/logs"
META_DIR="$JOB_DIR/metadata"
CKPT_DIR="$JOB_DIR/checkpoints"

# Create directories
mkdir -p "$JOB_DIR" "$LOGS_DIR" "$META_DIR" "$CKPT_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Experiment:    ${EXP_NAME:- (Unnamed)}"
echo "Job Directory: ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP
# ------------------------------------------------------------------
export WANDB_DIR="$LOGS_DIR"

WANDB_API_KEY=d5a57da73f0911bf27eb

# Load W&B Env file if mounted
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

# Configure W&B mode based on API key
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    
    # W&B Name Format: ID_NAME (e.g., 20362_vit_baseline)
    if [ -n "$EXP_NAME" ]; then
        export WANDB_NAME="${SLURM_JOB_ID}_${EXP_NAME}"
    else
        export WANDB_NAME="${SLURM_JOB_ID}"
    fi
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

# ------------------------------------------------------------------
# METADATA SAVING
# ------------------------------------------------------------------
echo "Job ID: ${SLURM_JOB_ID}" > "$META_DIR/experiment_metadata.txt"
echo "Experiment: ${EXP_NAME:-}" >> "$META_DIR/experiment_metadata.txt"
echo "Date: $(date)" >> "$META_DIR/experiment_metadata.txt"

if [ -f /home/baytas/Documents/Irem/tez_v2_clean/config.yaml ]; then
    cp /home/baytas/Documents/Irem/tez_v2_clean/config.yaml "$META_DIR/config_saved.yaml"
fi

# ------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------
echo "Starting training..."

cd "/home/baytas/Documents/Irem/tez_v2_clean"

# Run training
# Note: output.log is inside the ID folder, preventing any mix-up
python3.12 -u train.py \
    --config config.yaml 

#    --override \
#        debug.debug_mode=false \
#        logging.checkpoint_dir="$CKPT_DIR" \
#    2>&1 | tee "$LOGS_DIR/training_output.log"
#

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

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
