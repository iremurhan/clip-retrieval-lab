#!/bin/bash
# =============================================================================
# Figure 11 bucketed retrieval diagnostic.
#
# Builds caption length/complexity buckets if needed, reruns retrieval inference
# for all fresh WandB-matched checkpoints, then renders Figure 11.
#
# Usage:
#   sbatch scripts/eval/retrieval_by_bucket.slurm
# =============================================================================
#SBATCH --time=18:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/retrieval_by_bucket_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/retrieval_by_bucket_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace,CLIP_RETRIEVAL_ARTIFACT_ROOT=/output/results/artifacts/clip-retrieval-lab

set -euo pipefail

ARTIFACT_ROOT="/output/results/artifacts/clip-retrieval-lab"
JOB_DIR="/output/results/retrieval_by_bucket/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR" "$ARTIFACT_ROOT"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Artifact root: ${ARTIFACT_ROOT}"
echo "Job dir:       ${JOB_DIR}"
echo "=========================================="

export CLIP_RETRIEVAL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    echo "ERROR: WANDB_API_KEY not found; retrieval_by_bucket requires a fresh WandB API query."
    exit 1
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

echo ""
echo "=========================================="
echo "Bootstrapping runtime Python dependencies"
echo "=========================================="
if ! python3 -c "import pandas, matplotlib, seaborn" 2>/dev/null; then
    echo "[bootstrap] Installing pandas/matplotlib/seaborn..."
    pip install --quiet pandas matplotlib seaborn
fi
if ! python3 -c "import spacy" 2>/dev/null; then
    echo "[bootstrap] Installing spacy..."
    pip install --quiet spacy
fi
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "[bootstrap] Downloading spacy model en_core_web_sm..."
    python3 -m spacy download en_core_web_sm --quiet
fi

echo ""
echo "=========================================="
echo "Building caption bucket cache"
echo "=========================================="
python3 -u scripts/eval/build_caption_buckets.py \
    --data-root /workspace/datasets \
    --output-dir "${ARTIFACT_ROOT}/figs/cache"

echo ""
echo "=========================================="
echo "Running bucketed retrieval inference"
echo "=========================================="
python3 -u scripts/eval/retrieval_by_bucket.py \
    --results-root /output/results \
    --data-root /workspace/datasets \
    --device cuda \
    --batch-size 256 \
    --num-workers 8 \
    --force-output

echo ""
echo "=========================================="
echo "Rendering Figure 11"
echo "=========================================="
python3 -u figs/11_retrieval_by_bucket.py

SLURM_LOG="/users/beyza.urhan/experiments/results/retrieval_by_bucket_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    cp "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

echo ""
echo "Figure 11 retrieval-by-bucket pipeline finished."
echo "Outputs:"
echo "  ${ARTIFACT_ROOT}/figs/cache/retrieval_by_bucket.csv"
echo "  ${ARTIFACT_ROOT}/figs/cache/retrieval_by_bucket_wandb_manifest_snapshot.csv"
echo "  ${ARTIFACT_ROOT}/cache/retrieval_by_bucket_embeddings/"
echo "  ${ARTIFACT_ROOT}/figs/figures/11_retrieval_by_bucket_length.pdf"
echo "  ${ARTIFACT_ROOT}/figs/figures/11_retrieval_by_bucket_complexity.pdf"
