#!/bin/bash
# Usage: bash start_training_truba.sh <experiment_name> [config_path]
set -e

if [ -z "$1" ]; then
    echo "Usage: ./start_training_truba.sh <experiment_name> [config_path]"
    exit 1
fi

EXP_NAME="$1"
CONFIG="${2:-configs/config_coco.yaml}"

# Guess dataset name from config file
if [[ "$CONFIG" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi

# TRUBA Results Paths (Scratch)
RESULTS_ROOT="/arf/scratch/burhan/experiments/results"
# Log file path (using %j for SLURM Job ID)
SLURM_OUT="${RESULTS_ROOT}/${DATASET}/%j.out"

# Ensure parent log directory exists (SLURM won't create it)
mkdir -p "${RESULTS_ROOT}/${DATASET}"

echo "------------------------------------------------"
echo "Experiment: ${EXP_NAME}"
echo "Config:     ${CONFIG}"
echo "Log Path:   ${SLURM_OUT}"
echo "------------------------------------------------"

# Submit Job
# --output overrides the default in the slurm file.
# The variables ($EXP_NAME $CONFIG) at the end become $1 and $2 inside the slurm script.
JOB_ID=$(sbatch --job-name="${EXP_NAME}" \
    --output="${SLURM_OUT}" \
    --error="${SLURM_OUT}" \
    /arf/home/burhan/clip-retrieval-lab/scripts/truba/train_truba.slurm "$EXP_NAME" "$CONFIG" | awk '{print $NF}')

# Job ID is known now, create the result directory immediately to prevent wandb errors
mkdir -p "${RESULTS_ROOT}/${DATASET}/${JOB_ID}"

echo "Job Submitted! ID: ${JOB_ID}"
echo "To follow logs: tail -f ${RESULTS_ROOT}/${DATASET}/${JOB_ID}.out"
