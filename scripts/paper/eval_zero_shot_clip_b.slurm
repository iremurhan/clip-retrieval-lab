#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/paper_zero_shot_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/paper_zero_shot_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

RUN_ID="${1:?RUN_ID required: zero_shot_336}"
IMAGE_RESOLUTION="${2:?image resolution required}"
POSITIONAL_INTERPOLATION="${3:?positional interpolation flag required}"
SEED="${4:-42}"

REGISTRY="configs/registry_paper_clip_b.yaml"
CONFIG="configs/config_coco.yaml"
DATASET="coco"
BACKBONE_SLUG="clipb16"
PROJECT="cross-modal-retrieval-paper"
GROUP="baseline-calibration"

JOB_DIR="/output/results/paper/baseline-calibration/${BACKBONE_SLUG}/${RUN_ID}/r${IMAGE_RESOLUTION}/${DATASET}/s${SEED}/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"
OUT_JSON="${JOB_DIR}/zero_shot.json"
OUT_LOG="${JOB_DIR}/zero_shot.log"

case "$RUN_ID" in
    zero_shot_336) RUN_SUFFIX="zero-shot-r336_coco_s42" ;;
    *) echo "Unknown zero-shot RUN_ID: $RUN_ID (zero_shot_224 is intentionally not submitted for this stage)"; exit 1 ;;
esac
WANDB_RUN_NAME="${BACKBONE_SLUG}_${RUN_SUFFIX}"

echo "=========================================="
echo "Paper CLIP-B zero-shot evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run ID: ${RUN_ID}"
echo "Resolution: ${IMAGE_RESOLUTION}"
echo "Positional interpolation: ${POSITIONAL_INTERPOLATION}"
echo "Output: ${OUT_JSON}"
echo "W&B: ${PROJECT} / ${GROUP} / ${WANDB_RUN_NAME}"
echo "=========================================="

export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi
export WANDB_PROJECT="$PROJECT"

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

set +e
python3 -u tools/eval_paper_zero_shot.py \
    --config "$CONFIG" \
    --registry "$REGISTRY" \
    --run "$RUN_ID" \
    --split test \
    --seed "$SEED" \
    --output "$OUT_JSON" \
    --log-wandb \
    --wandb_project "$PROJECT" \
    --wandb_group "$GROUP" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    2>&1 | tee "$OUT_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

source /workspace/scripts/util/wandb_cleanup.sh
wandb_sync_and_cleanup "$JOB_DIR" "$PROJECT" "$EXIT_CODE"

SLURM_LOG="/users/beyza.urhan/experiments/results/paper_zero_shot_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "${JOB_DIR}/slurm.out"
fi

exit "$EXIT_CODE"
