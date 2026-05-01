#!/bin/bash
# =============================================================================
# SAM preprocessing for B5 variants.
#
# Usage:
#   # B5a/b/c masks, old behavior:
#   sbatch --array=0-7%4 scripts/preprocess/precompute_sam.slurm coco
#   sbatch --array=0-7%4 scripts/preprocess/precompute_sam.slurm flickr30k
#
#   # B5d SAM encoder features:
#   sbatch --array=0-7%4 scripts/preprocess/precompute_sam.slurm coco encoder_features 8
#   sbatch --array=0-7%4 scripts/preprocess/precompute_sam.slurm flickr30k encoder_features 8
#
#   # Merge B5d per-image feature shards after all array tasks finish:
#   sbatch scripts/preprocess/precompute_sam.slurm coco merge_encoder_features 8
#   sbatch scripts/preprocess/precompute_sam.slurm flickr30k merge_encoder_features 8
#
# The array index ($SLURM_ARRAY_TASK_ID) is forwarded as --shard_id and the
# array size as --num_shards, so 8 array tasks process disjoint slices of the
# image list. Output .npz files land alongside the source images, under
# /workspace/datasets/<dataset>/sam_masks/, so they live with the dataset (not
# inside the code checkout) and become visible to training automatically via
# the same bind mount used for images.
# Restartable: existing .npz files are skipped.
#
# Image: requires the SAM ViT-B checkpoint baked into /weights/ (see
# scripts/setup/Dockerfile).
# =============================================================================
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --container-image=biremurhan/image-text-contrast:b5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results
#SBATCH --output=/users/beyza.urhan/experiments/results/sam_precompute_%A_%a.out
#SBATCH --error=/users/beyza.urhan/experiments/results/sam_precompute_%A_%a.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch --array=0-N scripts/preprocess/precompute_sam.slurm <coco|flickr30k> [masks|encoder_features|merge_encoder_features] [pool_size]}"
MODE="${2:-masks}"
POOL_SIZE="${3:-8}"

case "$DATASET" in
    coco)
        IMAGE_DIR="/workspace/datasets/coco"
        MASK_OUTPUT_DIR="/workspace/datasets/coco/sam_masks"
        FEATURE_OUTPUT_DIR="/workspace/datasets/coco/sam_encoder_features"
        ;;
    flickr30k)
        IMAGE_DIR="/workspace/datasets/flickr30k"
        MASK_OUTPUT_DIR="/workspace/datasets/flickr30k/sam_masks"
        FEATURE_OUTPUT_DIR="/workspace/datasets/flickr30k/sam_encoder_features"
        ;;
    *)
        echo "Unknown dataset: $DATASET (expected coco or flickr30k)"
        exit 1
        ;;
esac

case "$MODE" in
    masks)
        OUTPUT_DIR="$MASK_OUTPUT_DIR"
        ;;
    encoder_features|merge_encoder_features)
        OUTPUT_DIR="$FEATURE_OUTPUT_DIR"
        ;;
    *)
        echo "Unknown mode: $MODE (expected masks, encoder_features, or merge_encoder_features)"
        exit 1
        ;;
esac

SAM_CKPT="/weights/sam_vit_b_01ec64.pth"
if [ "$MODE" != "merge_encoder_features" ] && [ ! -s "$SAM_CKPT" ]; then
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
echo "Mode:       ${MODE}"
echo "Pool size:  ${POOL_SIZE}"
echo "Shard:      ${SHARD_ID}/${NUM_SHARDS}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

if [ "$MODE" = "encoder_features" ]; then
    python3 -u scripts/preprocess/precompute_sam.py \
        --encoder_features \
        --dataset "$DATASET" \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --sam_checkpoint "$SAM_CKPT" \
        --pool_size "$POOL_SIZE" \
        --shard_id "$SHARD_ID" \
        --num_shards "$NUM_SHARDS"
elif [ "$MODE" = "merge_encoder_features" ]; then
    python3 -u scripts/preprocess/precompute_sam.py \
        --merge_encoder_features \
        --output_dir "$OUTPUT_DIR" \
        --pool_size "$POOL_SIZE"
else
    python3 -u scripts/preprocess/precompute_sam.py \
        --dataset "$DATASET" \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --sam_checkpoint "$SAM_CKPT" \
        --shard_id "$SHARD_ID" \
        --num_shards "$NUM_SHARDS"
fi
