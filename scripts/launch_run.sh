#!/usr/bin/env bash
# =============================================================================
# launch_run.sh — Batch SLURM submission for cross-modal retrieval experiments
# =============================================================================
# Submits one training job per (dataset, seed) pair. For COCO jobs, automatically
# chains a WiSE-FT evaluation job via --dependency=afterok.
#
# Usage:
#   bash scripts/launch_run.sh --run B0 [--datasets "coco flickr"] [--seeds "42 123 456"]
#
# Examples:
#   bash scripts/launch_run.sh --run B0
#   bash scripts/launch_run.sh --run FULL --datasets "coco" --seeds "42 123"
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
RUN_ID=""
DATASETS="coco flickr"
SEEDS="42 123 456"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)      RUN_ID="$2";    shift 2 ;;
        --datasets) DATASETS="$2";  shift 2 ;;
        --seeds)    SEEDS="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/launch_run.sh --run RUN_ID [--datasets \"coco flickr\"] [--seeds \"42 123 456\"]"
            exit 1
            ;;
    esac
done

if [ -z "$RUN_ID" ]; then
    echo "ERROR: --run RUN_ID is required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --------------------------------------------------------------------------
# Validate that required SLURM scripts exist
# --------------------------------------------------------------------------
TRAIN_SLURM="${SCRIPT_DIR}/train.slurm"
WISE_FT_SLURM="${SCRIPT_DIR}/wise_ft_job.slurm"

if [ ! -f "$TRAIN_SLURM" ]; then
    echo "ERROR: ${TRAIN_SLURM} not found."
    exit 1
fi
if [ ! -f "$WISE_FT_SLURM" ]; then
    echo "ERROR: ${WISE_FT_SLURM} not found."
    exit 1
fi

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
echo "========================================================"
echo "  launch_run.sh"
echo "  Run ID   : ${RUN_ID}"
echo "  Datasets : ${DATASETS}"
echo "  Seeds    : ${SEEDS}"
echo "========================================================"
printf "  %-20s  %-10s  %-6s  %-14s  %-14s\n" "RUN_NAME" "DATASET" "SEED" "TRAIN_JOB_ID" "WISE_JOB_ID"
printf "  %-20s  %-10s  %-6s  %-14s  %-14s\n" "--------------------" "----------" "------" "--------------" "--------------"

# --------------------------------------------------------------------------
# Submit jobs
# --------------------------------------------------------------------------
for DATASET in $DATASETS; do
    # Map dataset name to config file
    case "$DATASET" in
        coco)    CONFIG="configs/config_coco.yaml" ;;
        flickr)  CONFIG="configs/config_flickr.yaml" ;;
        *)
            echo "WARNING: Unknown dataset '${DATASET}'; skipping."
            continue
            ;;
    esac

    CONFIG_PATH="${REPO_ROOT}/${CONFIG}"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config not found: ${CONFIG_PATH}"
        exit 1
    fi

    for SEED in $SEEDS; do
        RUN_NAME="${RUN_ID}_${DATASET}_s${SEED}"

        # Submit training job
        # train.slurm positional args: <EXPERIMENT_NAME> <CONFIG_PATH>
        # Additional named args forwarded via --override in run.py
        TRAIN_JOB_ID=$(sbatch \
            --parsable \
            --job-name="${RUN_NAME}" \
            "$TRAIN_SLURM" \
            "$RUN_NAME" \
            "$CONFIG" \
            --override \
                "logging.run_id=${RUN_ID}" \
                "training.seed=${SEED}" \
        )

        CKPT_DIR="/output/results/${DATASET}/${TRAIN_JOB_ID}"

        WISE_JOB_ID=$(sbatch \
            --parsable \
            --job-name="${RUN_NAME}_wise" \
            --dependency="afterok:${TRAIN_JOB_ID}" \
            "$WISE_FT_SLURM" \
            --checkpoint "$CKPT_DIR" \
            --config     "$CONFIG" \
        )

        printf "  %-20s  %-10s  %-6s  %-14s  %-14s\n" \
            "$RUN_NAME" "$DATASET" "$SEED" "$TRAIN_JOB_ID" "$WISE_JOB_ID"
    done
done

echo "========================================================"
echo "All jobs submitted."
