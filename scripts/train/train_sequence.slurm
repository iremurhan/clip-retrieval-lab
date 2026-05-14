#!/bin/bash
# =============================================================================
# SLURM TRAINING SEQUENCE SCRIPT
# =============================================================================
# Runs several registry experiments sequentially inside one GPU allocation.
# Each inner run gets its own result/checkpoint directory and its own W&B run.
#
# Positional args:
#   $1  DATASET    coco | flickr | flickr30k
#   $2  SEED       integer seed
#   $3+ RUN_IDs    registry run IDs to run in order
#
# Usage:
#   sbatch scripts/train/train_sequence.slurm flickr 42 B0_projonly B0_uf1
# =============================================================================
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

DATASET_ARG="${1:?Usage: sbatch scripts/train/train_sequence.slurm <coco|flickr|flickr30k> <seed> <run_id> [run_id ...]}"
SEED="${2:?Usage: sbatch scripts/train/train_sequence.slurm <coco|flickr|flickr30k> <seed> <run_id> [run_id ...]}"
shift 2
RUN_IDS=("$@")

if [ ${#RUN_IDS[@]} -eq 0 ]; then
    echo "ERROR: at least one RUN_ID must be provided."
    exit 1
fi

case "$DATASET_ARG" in
    coco)
        DATASET="coco"
        CONFIG_FILE="configs/config_coco.yaml"
        ;;
    flickr|flickr30k)
        DATASET="flickr30k"
        CONFIG_FILE="configs/config_flickr30k.yaml"
        ;;
    *)
        echo "ERROR: unknown dataset '$DATASET_ARG' (expected coco, flickr, or flickr30k)."
        exit 1
        ;;
esac

WANDB_PROJECT_AUTO="clip-retrieval"
JOB_ROOT="/output/results/${DATASET}/${SLURM_JOB_ID}"
mkdir -p "$JOB_ROOT"

echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Dataset:    ${DATASET}"
echo "Config:     ${CONFIG_FILE}"
echo "Seed:       ${SEED}"
echo "Run IDs:    ${RUN_IDS[*]}"
echo "Job root:   ${JOB_ROOT}"
echo "=========================================="

if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    echo "W&B Project auto-set: $WANDB_PROJECT"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep parity with train.slurm for B2; harmless and idempotent for B0 variants.
if ! python3 -c "import spacy" 2>/dev/null; then
    echo "[bootstrap] Installing spacy..."
    pip install --quiet spacy
fi
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "[bootstrap] Downloading spacy model en_core_web_sm..."
    python3 -m spacy download en_core_web_sm --quiet
fi

source /workspace/scripts/util/wandb_cleanup.sh

FAILED_RUNS=()
RUN_INDEX=0
for RUN_ID in "${RUN_IDS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    RUN_DIR="${JOB_ROOT}/${RUN_INDEX}_${RUN_ID}_s${SEED}"
    mkdir -p "$RUN_DIR"
    export WANDB_DIR="$RUN_DIR"

    echo ""
    echo "=========================================="
    echo "Starting sequence item ${RUN_INDEX}/${#RUN_IDS[@]}"
    echo "Run ID:    ${RUN_ID}"
    echo "Run dir:   ${RUN_DIR}"
    echo "=========================================="

    set +e
    python3 -u run.py \
        --config "$CONFIG_FILE" \
        --run "$RUN_ID" \
        --seed "$SEED" \
        --override \
            logging.checkpoint_dir="$RUN_DIR" \
        > "$RUN_DIR/training.log" 2>&1
    TRAIN_EXIT_CODE=$?
    set -e

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "Run completed successfully: ${RUN_ID}"
    else
        echo "Run failed with exit code ${TRAIN_EXIT_CODE}: ${RUN_ID}"
        FAILED_RUNS+=("${RUN_ID}:${TRAIN_EXIT_CODE}")
    fi

    wandb_sync_and_cleanup "$RUN_DIR" "$WANDB_PROJECT_AUTO" "$TRAIN_EXIT_CODE"
done

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    cp "$SLURM_LOG" "$JOB_ROOT/slurm.out"
fi

echo ""
echo "=========================================="
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "All sequence runs completed successfully."
    echo "Results saved to: ${JOB_ROOT}"
    echo "=========================================="
    exit 0
fi

echo "Sequence finished with failures: ${FAILED_RUNS[*]}"
echo "Results saved to: ${JOB_ROOT}"
echo "=========================================="
exit 1
