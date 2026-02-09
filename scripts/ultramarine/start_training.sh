#!/bin/bash
# Submit training job. Usage: bash ./scripts/start_training.sh <run_name> [config_path]
# Example: bash ./scripts/start_training.sh "my_experiment" "configs/config_coco.yaml"
# Logs: results/{coco|flickr30k}/{job_id}/
set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/start_training.sh <run_name> [config_path]"
    echo "  run_name     - used for job name and W&B run name"
    echo "  config_path  - default: configs/config_coco.yaml"
    exit 1
fi

EXP_NAME="$1"
CONFIG="${2:-configs/config_coco.yaml}"

if [[ "$CONFIG" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi

RESULTS_ROOT="${EXPERIMENTS_RESULTS:-$HOME/experiments/results}"
SLURM_OUT="${RESULTS_ROOT}/${DATASET}/%j/slurm.out"

echo "------------------------------------------------"
echo "Run name:  ${EXP_NAME}"
echo "Config:    ${CONFIG}"
echo "Logs:      ${RESULTS_ROOT}/${DATASET}/<job_id>/slurm.out, training.log"
echo "------------------------------------------------"

JOB_ID=$(sbatch --job-name="${EXP_NAME}" \
    --output="${SLURM_OUT}" \
    --error="${SLURM_OUT}" \
    scripts/train.slurm "${EXP_NAME}" "${CONFIG}" | awk '{print $NF}')

mkdir -p "${RESULTS_ROOT}/${DATASET}/${JOB_ID}"
echo "Job ${JOB_ID} submitted. Logs: ${RESULTS_ROOT}/${DATASET}/${JOB_ID}/slurm.out, training.log"