#!/bin/bash
# =============================================================================
# B5_seg ONLY — Precompute SAM segment maps as a SLURM array job.
#
# Usage:
#   sbatch --array=0-7%4 scripts/precompute_sam.slurm coco
#   sbatch --array=0-7%4 scripts/precompute_sam.slurm flickr30k
#
# The array index ($SLURM_ARRAY_TASK_ID) is forwarded as --shard_id and the
# array size as --num_shards, so 8 array tasks process disjoint slices of the
# image list. Output .npz files land under /workspace/precomputed/sam_masks/.
# Restartable: existing .npz files are skipped.
#
# Image: requires the SAM ViT-B checkpoint baked into /weights/ (see
# scripts/setup/Dockerfile).
# =============================================================================
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.14
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results
#SBATCH --output=/users/beyza.urhan/experiments/results/sam_precompute_%A_%a.out
#SBATCH --error=/users/beyza.urhan/experiments/results/sam_precompute_%A_%a.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch --array=0-N scripts/precompute_sam.slurm <coco|flickr30k>}"

case "$DATASET" in
    coco)
        IMAGE_DIR="/workspace/datasets/coco"
        OUTPUT_DIR="/workspace/precomputed/sam_masks/coco"
        ;;
    flickr30k)
        IMAGE_DIR="/workspace/datasets/flickr30k"
        OUTPUT_DIR="/workspace/precomputed/sam_masks/flickr30k"
        ;;
    *)
        echo "Unknown dataset: $DATASET (expected coco or flickr30k)"
        exit 1
        ;;
esac

SAM_CKPT="/weights/sam_vit_b_01ec64.pth"
if [ ! -s "$SAM_CKPT" ]; then
    echo "FATAL: SAM checkpoint missing at $SAM_CKPT. Rebuild the container image."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "=========================================="
echo "Dataset:    ${DATASET}"
echo "Image dir:  ${IMAGE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "SAM ckpt:   ${SAM_CKPT}"
echo "Shard:      ${SHARD_ID}/${NUM_SHARDS}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

python3 -u scripts/precompute_sam.py \
    --dataset "$DATASET" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sam_checkpoint "$SAM_CKPT" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS"
