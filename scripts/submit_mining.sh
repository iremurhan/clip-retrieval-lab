#!/bin/bash
# Submit mining job. Usage: ./scripts/submit_mining.sh <modality> [config]
# Logs: results/{coco|flickr30k}/{job_id}/ (slurm.out, mining.log)
set -e

MODALITY="${1:-caption}"
CONFIG="${2:-configs/config_coco.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${EXPERIMENTS_RESULTS:-$HOME/experiments/results}"

if [[ "$CONFIG" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi

SLURM_OUT="${RESULTS_ROOT}/${DATASET}/%j/slurm.out"
mkdir -p "${RESULTS_ROOT}/${DATASET}"

JOB_ID=$(sbatch --export=MINING_MODALITY="$MODALITY",MINING_CONFIG="$CONFIG" \
    --output="${SLURM_OUT}" \
    --error="${SLURM_OUT}" \
    "$SCRIPT_DIR/mining.slurm" | awk '{print $NF}')

mkdir -p "${RESULTS_ROOT}/${DATASET}/${JOB_ID}"
echo "Job ${JOB_ID} submitted. Logs: ${RESULTS_ROOT}/${DATASET}/${JOB_ID}/"
