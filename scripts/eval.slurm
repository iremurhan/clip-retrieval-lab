#!/bin/bash
# =============================================================================
# SLURM EVALUATION SCRIPT — Normal + ECCV Modes
# =============================================================================
# Dispatched by: scripts/start_eval.sh (recommended)
# Direct usage:  sbatch scripts/eval.slurm <CONFIG_FILE> <CHECKPOINT_PATH> <SPLIT> <EVAL_MODE>
#
# EVAL_MODE:
#   normal  — Standard R@K / MAP@K via src.metrics
#   eccv    — ECCV Caption protocol via eccv_caption package
#
# Output directory: /output/results/{dataset}/eval_{SLURM_JOB_ID}/
#   ├── slurm.out      (SBATCH stdout/stderr)
#   ├── eval.log        (Python evaluation log)
#   └── wandb/          (WandB artifacts, synced then removed)
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
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

# =============================================================================
# Arguments
# =============================================================================
CONFIG_FILE="${1:-configs/config_coco.yaml}"
CHECKPOINT_PATH="${2:-}"
SPLIT="${3:-test}"
EVAL_MODE="${4:-normal}"

# --- Validate config ---
if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: /workspace/$CONFIG_FILE"
    exit 1
fi

# --- Validate checkpoint (if provided) ---
if [ -n "$CHECKPOINT_PATH" ] && [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# --- Validate eval mode ---
if [[ "$EVAL_MODE" != "normal" && "$EVAL_MODE" != "eccv" ]]; then
    echo "ERROR: EVAL_MODE must be 'normal' or 'eccv'. Got: $EVAL_MODE"
    exit 1
fi

# --- Derive dataset name ---
if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi

# --- WandB project: benchmark runs go to 'retrieval-benchmark', not 'retrieval' ---
if [[ "$EVAL_MODE" == "eccv" ]]; then
    WANDB_PROJECT_AUTO="retrieval-benchmark"
else
    WANDB_PROJECT_AUTO="retrieval-benchmark"
fi

# --- Job output directory (eval_ prefix prevents overwriting training logs) ---
JOB_DIR="/output/results/${DATASET}/eval_${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================="
echo "  DualEncoder Evaluation"
echo "=========================================="
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Config:       ${CONFIG_FILE}"
echo "  Checkpoint:   ${CHECKPOINT_PATH:-(zero-shot)}"
echo "  Split:        ${SPLIT}"
echo "  Mode:         ${EVAL_MODE}"
echo "  Dataset:      ${DATASET}"
echo "  WandB:        ${WANDB_PROJECT_AUTO}"
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
    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    export WANDB_NAME="${SLURM_JOB_ID}_${EVAL_MODE}_eval"
    echo "W&B: online mode, project=$WANDB_PROJECT"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

# =============================================================================
# Build and run evaluation command
# =============================================================================
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

CMD="python3 -u tools/eval.py"
CMD="$CMD --config $CONFIG_FILE"
CMD="$CMD --split $SPLIT"

# Checkpoint (optional — omit for zero-shot)
if [ -n "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT_PATH"
fi

# ECCV flag
if [[ "$EVAL_MODE" == "eccv" ]]; then
    CMD="$CMD --eccv"
fi

# Config overrides
CMD="$CMD --override"
CMD="$CMD logging.wandb_project=$WANDB_PROJECT_AUTO"
CMD="$CMD logging.use_wandb=true"

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
    echo "Evaluation FAILED with exit code: $EVAL_EXIT_CODE"
    echo "Check log: $JOB_DIR/eval.log"
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
    echo "WandB sync successful. Removing local cache."
    rm -rf "$JOB_DIR/wandb" 2>/dev/null || true
else
    echo "WARNING: WandB sync failed. Keeping '$JOB_DIR/wandb' for manual recovery."
fi

exit $EVAL_EXIT_CODE
