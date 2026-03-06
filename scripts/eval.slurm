#!/bin/bash
# =============================================================================
# SLURM EVALUATION SCRIPT — ECCV Caption Benchmark
# =============================================================================
# Dispatched by: scripts/start_eval.sh
# Direct:        sbatch --export=ALL,EVAL_CONFIG=...,EVAL_CHECKPOINT=... scripts/eval.slurm
#
# Env vars (set by start_eval.sh via sbatch --export):
#   EVAL_CONFIG      Config file path   (e.g. configs/config_coco.yaml)
#   EVAL_CHECKPOINT  .pth path or HF model ID  (e.g. openai/clip-vit-large-patch14-336)
#
# Output: /output/results/{dataset}/eval_{SLURM_JOB_ID}/
# =============================================================================
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/eval_%j/slurm.out
#SBATCH --error=/users/beyza.urhan/experiments/results/eval_%j/slurm.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

# =============================================================================
# Mount verification
# =============================================================================
echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output:    $(ls /output    2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL: Required mounts not found."
    exit 1
fi

# =============================================================================
# Read env vars (set by start_eval.sh via --export)
# =============================================================================
CONFIG_FILE="${EVAL_CONFIG:-configs/config_coco.yaml}"
CHECKPOINT="${EVAL_CHECKPOINT:-}"

# --- Validate config inside container ---
if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config not found: /workspace/$CONFIG_FILE"
    exit 1
fi

# --- Validate .pth checkpoint if applicable ---
if [[ "$CHECKPOINT" == *.pth && ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# --- Derive dataset name for output directory ---
if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi

# --- Job output directory (eval_ prefix prevents overwriting training logs) ---
JOB_DIR="/output/results/${DATASET}/eval_${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================="
echo "  ECCV Caption Benchmark Evaluation"
echo "=========================================="
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Config:       ${CONFIG_FILE}"
echo "  Checkpoint:   ${CHECKPOINT:-(zero-shot default)}"
echo "  Dataset:      ${DATASET}"
echo "  Job dir:      ${JOB_DIR}"
echo "=========================================="

# =============================================================================
# WandB setup
# =============================================================================
export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    export WANDB_PROJECT="retrieval-benchmark"
    echo "W&B: online, project=retrieval-benchmark"
else
    echo "W&B: offline (no API key)"
    export WANDB_MODE=offline
fi

# =============================================================================
# Install eccv_caption inside container
# =============================================================================
echo "Installing eccv_caption..."
pip install eccv_caption

# =============================================================================
# Run evaluation
# =============================================================================
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

CMD="python3 -u tools/eval.py --config $CONFIG_FILE"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

CMD="$CMD --override logging.wandb_project=retrieval-benchmark logging.use_wandb=true"

echo "Executing: $CMD"
eval $CMD > "$JOB_DIR/eval.log" 2>&1
EVAL_EXIT_CODE=$?

# =============================================================================
# Post-run
# =============================================================================
echo ""
echo "=========================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation FAILED (exit code: $EVAL_EXIT_CODE)"
    echo "Check: $JOB_DIR/eval.log"
fi
echo "=========================================="

# --- WandB sync & cleanup ---
SYNC_EXIT=1
if [ -d "$JOB_DIR/wandb" ]; then
    echo "Syncing WandB logs..."
    python3 -m wandb sync "$JOB_DIR/wandb/latest-run" > "$JOB_DIR/wandb_sync.log" 2>&1
    SYNC_EXIT=$?
fi

if [ $SYNC_EXIT -eq 0 ]; then
    echo "WandB sync OK. Removing local cache."
    rm -rf "$JOB_DIR/wandb" 2>/dev/null || true
else
    echo "WARNING: WandB sync failed. Keeping local cache at $JOB_DIR/wandb"
fi

exit $EVAL_EXIT_CODE
