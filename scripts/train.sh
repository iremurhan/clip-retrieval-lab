#!/bin/bash

#SBATCH -p normal

#SBATCH --time=4-00:00:00

#SBATCH --account=root

#SBATCH --job-name=irem_tez

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1



# --- LOG DIRECTORIES (Outside scripts folder) ---

# Outputs will be saved under tez_v2_clean/slurm_logs

#SBATCH --output=/home/baytas/Documents/Irem/tez_v2_clean/slurm_logs/out-%j.out

#SBATCH --error=/home/baytas/Documents/Irem/tez_v2_clean/slurm_logs/err-%j.err



# --- Environment Setup ---

set -euo pipefail



# Module loading disabled (using system python)

# module load Python/3.12.0 cuda/10.1 


# ------------------------------------------------------------------

# DIRECTORIES

# ------------------------------------------------------------------

# Project root directory

PROJECT_ROOT="/home/baytas/Documents/Irem/tez_v2_clean"

cd "$PROJECT_ROOT"



# Get SLURM Job ID

JOB_ID=${SLURM_JOB_ID}

EXP_NAME="${1:-}"



# Main directory for checkpoints and training outputs

JOB_DIR="$PROJECT_ROOT/runs/${JOB_ID}"



# Subdirectories

LOGS_DIR="$JOB_DIR/logs"

META_DIR="$JOB_DIR/metadata"

CKPT_DIR="$JOB_DIR/checkpoints"



# Create directories

mkdir -p "$LOGS_DIR" "$META_DIR" "$CKPT_DIR"



echo "=========================================="

echo "Job ID:        ${JOB_ID}"

echo "Experiment:    ${EXP_NAME:- (Unnamed)}"

echo "Job Directory: ${JOB_DIR}"

echo "Log Directory: $PROJECT_ROOT/slurm_logs"

echo "=========================================="



# ------------------------------------------------------------------

# W&B SETUP (KEY ADDED)

# ------------------------------------------------------------------

export WANDB_DIR="$LOGS_DIR"



# Added W&B API Key here. No login required anymore.

export WANDB_MODE=online



# Determine project name

if [ -n "$EXP_NAME" ]; then

    export WANDB_NAME="${JOB_ID}_${EXP_NAME}"

else

    export WANDB_NAME="${JOB_ID}"

fi



# ------------------------------------------------------------------

# METADATA SAVING

# ------------------------------------------------------------------

echo "Job ID: ${JOB_ID}" > "$META_DIR/experiment_metadata.txt"

echo "Date: $(date)" >> "$META_DIR/experiment_metadata.txt"

# Backup config file

cp config.yaml "$META_DIR/config_saved.yaml"



# ------------------------------------------------------------------

# START TRAINING

# ------------------------------------------------------------------

echo "Starting training..."

source /home/baytas/Documents/Irem/tez_v2_clean/venv/bin/activate

# Start with python3

python3 -u train.py \

    --config config.yaml \

    --override \

        logging.checkpoint_dir="$CKPT_DIR" \

        logging.wandb_project="tez-slurm-run"



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
