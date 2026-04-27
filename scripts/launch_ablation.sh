#!/bin/bash
# =============================================================================
# ABLATION LAUNCHER
# =============================================================================
# Submits a single training job for a given run ID, seed, and config.
# For COCO jobs, also submits a dependent ECCV evaluation job.
#
# Usage:
#   bash scripts/launch_ablation.sh --run <RUN_ID> --seed <SEED> --config <CONFIG> [-- KEY=VAL ...]
#
# Example:
#   bash scripts/launch_ablation.sh --run B0 --seed 42 --config configs/config_flickr30k.yaml
#
# Optional trailing overrides (after `--`) are passed verbatim to run.py's --override:
#   bash scripts/launch_ablation.sh --run B0v2 --seed 42 --config configs/config_flickr30k.yaml \
#       -- paraphraser.type=llama
#
# All three named arguments are required.
#
# Checkpoint paths (inside container):
#   /output/results/{dataset}/{TRAIN_JOB_ID}/best_model.pth
# =============================================================================
set -euo pipefail

# --- Parse arguments ---
RUN_ID=""
SEED=""
CONFIG=""
EXTRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_ID="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_OVERRIDES=("$@")
            break
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Usage: bash scripts/launch_ablation.sh --run <RUN_ID> --seed <SEED> --config <CONFIG> [-- KEY=VAL ...]"
            exit 1
            ;;
    esac
done

if [ -z "$RUN_ID" ] || [ -z "$SEED" ] || [ -z "$CONFIG" ]; then
    echo "ERROR: --run, --seed, and --config are all required."
    echo "Usage: bash scripts/launch_ablation.sh --run <RUN_ID> --seed <SEED> --config <CONFIG> [-- KEY=VAL ...]"
    exit 1
fi

# Validate extra overrides look like KEY=VAL
for ov in "${EXTRA_OVERRIDES[@]}"; do
    if [[ "$ov" != *=* ]]; then
        echo "ERROR: Override '$ov' is not of the form KEY=VAL."
        exit 1
    fi
done

# Infer dataset name from config filename (config_coco.yaml -> coco)
DATASET=$(basename "$CONFIG" .yaml | sed 's/^config_//')

RESULTS_ROOT="${EXPERIMENTS_RESULTS:-$HOME/experiments/results}"

echo "=========================================="
echo "  Ablation Launcher"
echo "  Run ID:   $RUN_ID"
echo "  Seed:     $SEED"
echo "  Dataset:  $DATASET"
echo "  Config:   $CONFIG"
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
    echo "  Extra overrides:"
    for ov in "${EXTRA_OVERRIDES[@]}"; do
        echo "    - $ov"
    done
fi
echo "=========================================="

# Submit training job
TRAIN_JOB_ID=$(sbatch \
    --job-name="${RUN_ID}_${DATASET}_s${SEED}" \
    --output="${RESULTS_ROOT}/%j/slurm.out" \
    --error="${RESULTS_ROOT}/%j/slurm.out" \
    scripts/ablation_train.slurm \
        "$RUN_ID" "$SEED" "$CONFIG" "${EXTRA_OVERRIDES[@]}" \
    | awk '{print $NF}')

echo "  [TRAIN] ${DATASET} seed=${SEED}  job_id=${TRAIN_JOB_ID}"

echo ""
echo "=========================================="
echo "Job submitted. Monitor with:  squeue -u \$USER"
echo "=========================================="
