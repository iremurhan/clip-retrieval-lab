#!/bin/bash
# =============================================================================
# SLURM TRAINING SCRIPT (unified)
# =============================================================================
# Single training entry point for all experiment types. Replaces the former
# train.slurm and ablation_train.slurm, which were near-identical.
#
# Positional args:
#   $1  RUN_ID            Registry run ID (e.g. B0, B0plus, B2) — passed to --run
#   $2  CONFIG_PATH       Path to dataset config, relative to /workspace
#                         (e.g. "configs/config_coco.yaml")
#   $3+ EXTRA_ARGS        Any additional flags forwarded verbatim to run.py
#                         (e.g. --seed 42, --override key=value ...)
#
# Usage (direct sbatch):
#   sbatch scripts/train/train.slurm <RUN_ID> <CONFIG_PATH> [run.py flags...]
#
# Examples:
#   # Single run with registry overrides and seed
#   sbatch scripts/train/train.slurm B0 configs/config_coco.yaml --seed 42
#
#   # With extra overrides
#   sbatch scripts/train/train.slurm B0plus configs/config_flickr30k.yaml \
#       --seed 123 --override paraphraser.type=llama
#
#   # Debug run (no W&B, fast)
#   sbatch scripts/train/train.slurm B0 configs/config_coco.yaml \
#       --seed 42 --override debug.debug_mode=true
#
# Note: Prefer launch_run.sh for batch submission across datasets and seeds.
#
# Outputs:
#   /output/results/{dataset}/{job_id}/slurm.out       SLURM stdout+stderr
#   /output/results/{dataset}/{job_id}/training.log    Python stdout
#   /output/results/{dataset}/{job_id}/best_model.pth  Best checkpoint
# =============================================================================
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

# --- Mount validation ---
echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

# --- Parse positional args ---
RUN_ID="${1:-unnamed}"
CONFIG_FILE="${2:-configs/config_coco.yaml}"
shift 2
EXTRA_ARGS=("$@")

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# --- Detect dataset ---
source /workspace/scripts/util/detect_dataset.sh
DATASET=$(detect_dataset "$CONFIG_FILE") || exit 1

WANDB_PROJECT_AUTO="clip-retrieval"
JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
CKPT_DIR="$JOB_DIR"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Run ID:     ${RUN_ID}"
echo "Dataset:    ${DATASET}"
echo "Config:     ${CONFIG_FILE}"
echo "Job dir:    ${JOB_DIR}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra args: ${EXTRA_ARGS[*]}"
fi
echo "=========================================="

# --- WandB setup ---
export WANDB_DIR="$JOB_DIR"
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

# --- Runtime dependency bootstrap ---
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# spaCy bootstrap for B2 (POS-tag hard-neg generator). Idempotent — installs
# only if `import spacy` or `spacy.load('en_core_web_sm')` fails.
# TODO: bake `spacy + en_core_web_sm` into the next container image tag and
# drop this block.
if ! python3 -c "import spacy" 2>/dev/null; then
    echo "[bootstrap] Installing spacy..."
    pip install --quiet spacy
fi
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "[bootstrap] Downloading spacy model en_core_web_sm..."
    python3 -m spacy download en_core_web_sm --quiet
fi

# --- Training ---
echo "Starting training..."

set +e
python3 -u run.py \
    --config "$CONFIG_FILE" \
    --run "$RUN_ID" \
    --override \
        logging.checkpoint_dir="$CKPT_DIR" \
    "${EXTRA_ARGS[@]}" \
    > "$JOB_DIR/training.log" 2>&1
TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Results saved to: $JOB_DIR"
echo "=========================================="

# --- WandB sync + verify + cleanup ---
source /workspace/scripts/util/wandb_cleanup.sh
wandb_sync_and_cleanup "$JOB_DIR" "$WANDB_PROJECT_AUTO" "$TRAIN_EXIT_CODE"

# --- Move SLURM log into per-job directory ---
SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $TRAIN_EXIT_CODE
