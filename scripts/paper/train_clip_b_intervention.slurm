#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/paper_intervention_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/paper_intervention_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

RUN_ID="${1:?RUN_ID required}"
SEED="${2:-42}"

REGISTRY="configs/registry_paper_clip_b.yaml"
CONFIG="configs/config_coco.yaml"
DATASET="coco"
IMAGE_RESOLUTION="336"
BACKBONE_SLUG="clipb16"
PROJECT="cross-modal-retrieval-paper"

JOB_DIR="/output/results/paper/interventions/${BACKBONE_SLUG}/${RUN_ID}/r${IMAGE_RESOLUTION}/${DATASET}/s${SEED}/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Paper CLIP-B intervention training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run ID: ${RUN_ID}"
echo "Selected unfreeze depth: 1"
echo "Output: ${JOB_DIR}"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$RUN_ID" = "B2" ]; then
    if ! python3 -c "import spacy" 2>/dev/null; then
        echo "[bootstrap] Installing spacy..."
        pip install --quiet spacy
    fi
    if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
        echo "[bootstrap] Downloading spacy model en_core_web_sm..."
        python3 -m spacy download en_core_web_sm --quiet
    fi
fi

set +e
python3 -u run.py \
    --config "$CONFIG" \
    --registry "$REGISTRY" \
    --run "$RUN_ID" \
    --seed "$SEED" \
    --override \
        logging.checkpoint_dir="$JOB_DIR" \
    > "$JOB_DIR/training.log" 2>&1
EXIT_CODE=$?
set -e

source /workspace/scripts/util/wandb_cleanup.sh
wandb_sync_and_cleanup "$JOB_DIR" "$PROJECT" "$EXIT_CODE"

SLURM_LOG="/users/beyza.urhan/experiments/results/paper_intervention_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "${JOB_DIR}/slurm.out"
fi

exit "$EXIT_CODE"
