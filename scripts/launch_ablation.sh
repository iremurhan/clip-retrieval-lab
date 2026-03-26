#!/bin/bash
# =============================================================================
# ABLATION LAUNCHER
# =============================================================================
# Submits one training job per seed x dataset combination for a given run ID.
# For COCO jobs, also submits a dependent ECCV evaluation job.
#
# Usage:
#   bash scripts/launch_ablation.sh --run <RUN_ID>
#
# Example:
#   bash scripts/launch_ablation.sh --run B0
#
# Jobs submitted per --run:
#   coco    x seeds [42, 123, 456] = 3 training + 3 ECCV eval (dependent)
#   flickr  x seeds [42, 123, 456] = 3 training
#   Total: 9 jobs
#
# Checkpoint paths (inside container):
#   /output/results/{dataset}/{TRAIN_JOB_ID}/best_model.pth
# =============================================================================
set -euo pipefail

SEEDS=(42 123 456)
CONFIGS=(
    "coco:configs/config_coco.yaml"
    "flickr30k:configs/config_flickr30k.yaml"
)

# --- Parse arguments ---
RUN_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Usage: bash scripts/launch_ablation.sh --run <RUN_ID>"
            exit 1
            ;;
    esac
done

if [ -z "$RUN_ID" ]; then
    echo "ERROR: --run is required."
    echo "Usage: bash scripts/launch_ablation.sh --run <RUN_ID>"
    exit 1
fi

RESULTS_ROOT="${EXPERIMENTS_RESULTS:-$HOME/experiments/results}"

echo "=========================================="
echo "  Ablation Launcher"
echo "  Run ID:  $RUN_ID"
echo "  Seeds:   ${SEEDS[*]}"
echo "=========================================="

for ENTRY in "${CONFIGS[@]}"; do
    DATASET="${ENTRY%%:*}"
    CONFIG="${ENTRY##*:}"

    echo ""
    echo "--- Dataset: $DATASET ---"

    for SEED in "${SEEDS[@]}"; do
        # Submit training job
        TRAIN_JOB_ID=$(sbatch \
            --job-name="${RUN_ID}_${DATASET}_s${SEED}" \
            --output="${RESULTS_ROOT}/%j/slurm.out" \
            --error="${RESULTS_ROOT}/%j/slurm.out" \
            scripts/ablation_train.slurm \
                "$RUN_ID" "$SEED" "$CONFIG" \
            | awk '{print $NF}')

        echo "  [TRAIN] ${DATASET} seed=${SEED}  job_id=${TRAIN_JOB_ID}"

        # For COCO: submit dependent ECCV eval job
        if [ "$DATASET" = "coco" ]; then
            # Checkpoint is written to /output/results/coco/{TRAIN_JOB_ID}/best_model.pth
            # inside the container (mounted from $RESULTS_ROOT/coco/{TRAIN_JOB_ID}/)
            CKPT_CONTAINER="/output/results/coco/${TRAIN_JOB_ID}/best_model.pth"

            EVAL_JOB_ID=$(sbatch \
                --job-name="eval_${RUN_ID}_s${SEED}" \
                --dependency="afterok:${TRAIN_JOB_ID}" \
                scripts/eval.slurm \
                    "$CONFIG" "$CKPT_CONTAINER" "test" "eccv" \
                | awk '{print $NF}')

            echo "  [EVAL]  ${DATASET} seed=${SEED}  job_id=${EVAL_JOB_ID}  (depends on ${TRAIN_JOB_ID})"
        fi
    done
done

echo ""
echo "=========================================="
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
echo "=========================================="
