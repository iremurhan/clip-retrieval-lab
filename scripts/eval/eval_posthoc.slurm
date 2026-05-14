#!/bin/bash
# =============================================================================
# Post-hoc evaluation bundle for completed checkpoints.
# =============================================================================
# Runs SugarCrepe, cross-dataset OOD retrieval, and MMVP-VLM for one checkpoint.
# Each evaluation is isolated: a failure is logged but does not prevent the
# remaining evaluations from running.
#
# Positional args:
#   $1  CKPT_PATH       Path to best_model.pth checkpoint
#   $2  WANDB_RUN_ID    Optional W&B run ID to resume and log summary metrics
#   $3  WANDB_RUN_NAME  Optional W&B run name for run/dataset/seed parsing
#
# Usage:
#   sbatch scripts/eval/eval_posthoc.slurm \
#       /output/results/flickr30k/B0plus_fixed_flickr30k_s42/best_model.pth \
#       25p8m9ym \
#       B0plus_fixed_flickr30k_s42
# =============================================================================
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
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

JOB_DIR="/output/results/posthoc_eval/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Checkpoint:    ${CKPT_PATH}"
echo "Output dir:    ${JOB_DIR}"
echo "W&B run ID:    ${WANDB_RUN_ID:-(none)}"
echo "W&B run name:  ${WANDB_RUN_NAME:-(none)}"
echo "=========================================="

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

COMMON_WANDB_ARGS=()
if [ -n "$WANDB_RUN_ID" ]; then
    COMMON_WANDB_ARGS+=(--wandb_run_id "$WANDB_RUN_ID")
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    COMMON_WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

run_eval() {
    local name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "Starting ${name}"
    echo "=========================================="
    set +e
    "$@" > "$JOB_DIR/${name}.log" 2>&1
    local exit_code=$?
    set -e
    echo "${name} exit_code=${exit_code}" | tee -a "$JOB_DIR/status.log"
    return 0
}

SUGAR_WANDB_ARGS=()
if [ -n "$WANDB_RUN_ID" ]; then
    SUGAR_WANDB_ARGS+=(--wandb_run_id "$WANDB_RUN_ID")
fi

run_eval sugarcrepe \
    python3 -u scripts/eval/eval_sugarcrepe.py \
        --checkpoint "$CKPT_PATH" \
        --data_dir datasets/sugarcrepe \
        --images_dir datasets/coco/val2017 \
        "${SUGAR_WANDB_ARGS[@]}"

OOD_ARGS=(--checkpoint "$CKPT_PATH" --data-root datasets --cache-dir "$JOB_DIR/ood_eval" --force)
if [ -n "$WANDB_RUN_ID" ]; then
    OOD_ARGS+=(--log-wandb --wandb_run_id "$WANDB_RUN_ID")
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    OOD_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
run_eval ood_retrieval python3 -u scripts/eval/eval_cross_dataset.py "${OOD_ARGS[@]}"

MMVP_ARGS=(--checkpoint "$CKPT_PATH" --data-dir datasets/mmvp_vlm --cache-dir "$JOB_DIR/mmvp_vlm" --force)
if [ -n "$WANDB_RUN_ID" ]; then
    MMVP_ARGS+=(--log-wandb --wandb_run_id "$WANDB_RUN_ID")
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    MMVP_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
run_eval mmvp_vlm python3 -u scripts/eval/eval_mmvp_vlm.py "${MMVP_ARGS[@]}"

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

echo ""
echo "Post-hoc eval bundle finished. Logs: ${JOB_DIR}"
