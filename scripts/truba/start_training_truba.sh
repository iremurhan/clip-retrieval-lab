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

# Ensure parent log directory exists (SLURM won't create it)
mkdir -p "${RESULTS_ROOT}/${DATASET}"
LOG_FILE="${RESULTS_ROOT}/${DATASET}/%j.out"

echo "------------------------------------------------"
echo "Experiment: ${EXP_NAME}"
echo "Config:     ${CONFIG}"
echo "Log:        ${RESULTS_ROOT}/${DATASET}/<JOB_ID>.out"
echo "------------------------------------------------"

# Submit Job
sbatch --job-name="${EXP_NAME}" \
       --output="${LOG_FILE}" \
       --error="${LOG_FILE}" \
       /arf/home/burhan/clip-retrieval-lab/scripts/truba/train_truba.slurm "$EXP_NAME" "$CONFIG"
