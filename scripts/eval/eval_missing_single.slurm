#!/bin/bash
# =============================================================================
# Strict single-evaluation backfill for completed checkpoints.
# =============================================================================
# Positional args:
#   $1  EVAL_TYPE       sugarcrepe|mmvp_vlm|ood_retrieval|in_domain_retrieval
#   $2  CKPT_PATH       Path to best_model.pth checkpoint inside the container
#   $3  WANDB_RUN_ID    W&B run ID to resume and log summary metrics
#   $4  WANDB_RUN_NAME  Optional W&B run name for run/dataset/seed parsing
#   $5  TRAIN_DATASET   Optional train dataset for OOD/in-domain routing
#   $6  EVAL_DATASET    Optional eval dataset for OOD/in-domain routing
#
# Usage:
#   sbatch scripts/eval/eval_missing_single.slurm \
#       sugarcrepe /output/results/coco/123/best_model.pth abc123 B0_coco_s42 coco coco
# =============================================================================
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_missing_eval.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_missing_eval.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

EVAL_TYPE="${1:?ERROR: eval type required as \$1}"
CKPT_PATH="${2:?ERROR: checkpoint path required as \$2}"
WANDB_RUN_ID="${3:?ERROR: W&B run id required as \$3}"
WANDB_RUN_NAME="${4:-}"
TRAIN_DATASET="${5:-}"
EVAL_DATASET="${6:-}"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

JOB_DIR="/output/results/missing_eval_backfill/jobs/${SLURM_JOB_ID}_${EVAL_TYPE}"
mkdir -p "$JOB_DIR"

export WANDB_DIR="$JOB_DIR/wandb"
mkdir -p "$WANDB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not available; refusing to run an eval that cannot log online."
    exit 2
fi
export WANDB_MODE=online

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

COMMON_WANDB_ARGS=(--wandb_run_id "$WANDB_RUN_ID")
if [ -n "$WANDB_RUN_NAME" ]; then
    COMMON_WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Eval type:     ${EVAL_TYPE}"
echo "Checkpoint:    ${CKPT_PATH}"
echo "Output dir:    ${JOB_DIR}"
echo "W&B run ID:    ${WANDB_RUN_ID}"
echo "W&B run name:  ${WANDB_RUN_NAME:-(none)}"
echo "Train dataset: ${TRAIN_DATASET:-(auto)}"
echo "Eval dataset:  ${EVAL_DATASET:-(auto)}"
echo "=========================================="

case "$EVAL_TYPE" in
    sugarcrepe)
        python3 -u scripts/eval/eval_sugarcrepe.py \
            --checkpoint "$CKPT_PATH" \
            --data_dir datasets/sugarcrepe \
            --images_dir datasets/coco/val2017 \
            --wandb_run_id "$WANDB_RUN_ID" \
            > "$JOB_DIR/eval.log" 2>&1
        ;;

    mmvp_vlm)
        python3 -u scripts/eval/eval_mmvp_vlm.py \
            --checkpoint "$CKPT_PATH" \
            --data-dir datasets/mmvp_vlm \
            --cache-dir "$JOB_DIR/mmvp_vlm" \
            --output "$JOB_DIR/mmvp_vlm.json" \
            --force \
            --log-wandb \
            "${COMMON_WANDB_ARGS[@]}" \
            > "$JOB_DIR/eval.log" 2>&1
        ;;

    ood_retrieval)
        OOD_ARGS=(
            --checkpoint "$CKPT_PATH"
            --data-root datasets
            --cache-dir "$JOB_DIR/ood_eval"
            --output "$JOB_DIR/ood_retrieval.json"
            --force
            --log-wandb
            "${COMMON_WANDB_ARGS[@]}"
        )
        if [ -n "$TRAIN_DATASET" ]; then
            OOD_ARGS+=(--train-dataset "$TRAIN_DATASET")
        fi
        if [ -n "$EVAL_DATASET" ]; then
            OOD_ARGS+=(--eval-dataset "$EVAL_DATASET")
        fi
        python3 -u scripts/eval/eval_cross_dataset.py "${OOD_ARGS[@]}" \
            > "$JOB_DIR/eval.log" 2>&1
        ;;

    in_domain_retrieval)
        IN_DOMAIN_ARGS=(
            --checkpoint "$CKPT_PATH"
            --data-root datasets
            --output "$JOB_DIR/in_domain_retrieval.json"
            --log-wandb
            "${COMMON_WANDB_ARGS[@]}"
        )
        if [ -n "$EVAL_DATASET" ]; then
            IN_DOMAIN_ARGS+=(--dataset "$EVAL_DATASET")
        elif [ -n "$TRAIN_DATASET" ]; then
            IN_DOMAIN_ARGS+=(--dataset "$TRAIN_DATASET")
        fi
        python3 -u scripts/eval/eval_in_domain_retrieval.py "${IN_DOMAIN_ARGS[@]}" \
            > "$JOB_DIR/eval.log" 2>&1
        ;;

    *)
        echo "ERROR: Unsupported EVAL_TYPE: $EVAL_TYPE"
        exit 64
        ;;
esac

echo "Evaluation completed successfully. Logs: $JOB_DIR"
