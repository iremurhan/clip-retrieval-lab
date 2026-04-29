#!/bin/bash
# =============================================================================
# Extract worst retrieval failures from COCO 5K test set.
# =============================================================================
# Positional args:
#   $1  CKPT_PATH       Path to best_model.pth checkpoint
#   $2  CONFIG_PATH     Path to dataset config (default: configs/config_coco.yaml)
#   $3  WANDB_RUN_ID    (optional) WandB run ID to resume and log results
#
# Usage:
#   sbatch scripts/eval/extract_failures.slurm \
#       /output/results/coco/12345/best_model.pth \
#       configs/config_coco.yaml \
#       abc123def
#
# Examples:
#   # COCO failures with default config (no W&B)
#   sbatch scripts/eval/extract_failures.slurm \
#       /output/results/coco/12345/best_model.pth
#
#   # Flickr30k failures with explicit config
#   sbatch scripts/eval/extract_failures.slurm \
#       /output/results/flickr/67890/best_model.pth \
#       configs/config_flickr30k.yaml
#
#   # COCO failures, log to existing W&B run
#   sbatch scripts/eval/extract_failures.slurm \
#       /output/results/coco/12345/best_model.pth \
#       configs/config_coco.yaml \
#       abc123def
#
# Outputs:
#   /output/results/failure_analysis/{job_id}/failures.html
#   /output/results/failure_analysis/{job_id}/failures.json
#   /output/results/failure_analysis/{job_id}/eval.log
# =============================================================================
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:b5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
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

CKPT_PATH="${1:?ERROR: checkpoint path required as \$1}"
CONFIG_FILE="${2:-configs/config_coco.yaml}"
WANDB_RUN_ID="${3:-}"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: /workspace/$CONFIG_FILE"
    exit 1
fi

JOB_DIR="/output/results/failure_analysis/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Checkpoint:  ${CKPT_PATH}"
echo "Config:      ${CONFIG_FILE}"
echo "Output dir:  ${JOB_DIR}"
echo "W&B run ID:  ${WANDB_RUN_ID:-(none)}"
echo "=========================================="

# W&B setup
export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    echo "W&B API Key not found. WandB logging disabled."
    export WANDB_MODE=offline
    WANDB_RUN_ID=""
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

WANDB_ARGS=""
if [ -n "$WANDB_RUN_ID" ]; then
    WANDB_ARGS="--wandb_run_id $WANDB_RUN_ID"
fi

python3 -u scripts/eval/extract_failures.py \
    --checkpoint "$CKPT_PATH" \
    --config "$CONFIG_FILE" \
    --output_dir "$JOB_DIR" \
    $WANDB_ARGS \
    > "$JOB_DIR/eval.log" 2>&1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Failure extraction completed successfully!"
    echo "HTML report: $JOB_DIR/failures.html"
    echo "JSON data:   $JOB_DIR/failures.json"
else
    echo "Failure extraction failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $EXIT_CODE
