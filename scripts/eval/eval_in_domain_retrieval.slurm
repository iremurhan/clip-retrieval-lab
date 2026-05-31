#!/bin/bash
# =============================================================================
# Standalone in-domain retrieval backfill for completed checkpoints.
# =============================================================================
# Positional args:
#   $1  CKPT_PATH       Path to best_model.pth checkpoint
#   $2  WANDB_RUN_ID    Optional W&B run ID to resume and log summary metrics
#   $3  WANDB_RUN_NAME  Optional W&B run name for run/dataset/seed parsing
#
# Usage:
#   sbatch scripts/eval/eval_in_domain_retrieval.slurm \
#       /output/results/coco/12345/best_model.pth \
#       abc123def \
#       B0_coco_s42
# =============================================================================
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

CKPT_PATH="${1:?ERROR: checkpoint path required as \$1}"
WANDB_RUN_ID="${2:-}"
WANDB_RUN_NAME="${3:-}"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

JOB_DIR="/output/results/in_domain_eval/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ] && [ -n "$WANDB_RUN_ID" ]; then
    export WANDB_MODE=online
    WANDB_ARGS+=(--log-wandb --wandb_run_id "$WANDB_RUN_ID")
else
    export WANDB_MODE=offline
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

python3 -u scripts/eval/eval_in_domain_retrieval.py \
    --checkpoint "$CKPT_PATH" \
    --data-root datasets \
    --output "$JOB_DIR/in_domain_retrieval.json" \
    "${WANDB_ARGS[@]}" \
    > "$JOB_DIR/eval.log" 2>&1

EXIT_CODE=$?

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

echo "In-domain retrieval eval finished with exit_code=${EXIT_CODE}. Logs: ${JOB_DIR}"
exit $EXIT_CODE
