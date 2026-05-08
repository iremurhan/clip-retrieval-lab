#!/bin/bash
# =============================================================================
# Audit caption_rewrites coverage for a (dataset, model) pair: report K-distribution
# under the normalized-dedup semantics, and optionally dump K<2 failures to a
# JSONL for the retry pass.
#
# Usage:
#   sbatch scripts/setup/check_rewrite_coverage.slurm <dataset> <model_slug> [dump_failed:0|1]
#
# Examples:
#   sbatch scripts/setup/check_rewrite_coverage.slurm flickr30k meta-llama-3-8b-instruct
#   sbatch scripts/setup/check_rewrite_coverage.slurm flickr30k mistral-7b-instruct-v0.2 1
#   sbatch scripts/setup/check_rewrite_coverage.slurm coco      meta-llama-3-8b-instruct
# =============================================================================
#SBATCH --time=00:15:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_coverage.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_coverage.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch <script> <coco|flickr30k> <model_slug> [dump_failed:0|1]}"
MODEL_SLUG="${2:?Usage: sbatch <script> <dataset> <model_slug> [dump_failed:0|1]}"
DUMP_FAILED="${3:-0}"

case "$DATASET" in
    coco)
        KARPATHY="/workspace/datasets/coco/caption_datasets/dataset_coco.json"
        REWRITES="/workspace/datasets/coco/caption_rewrites_${MODEL_SLUG}.json"
        DUMP_PATH="/workspace/datasets/coco/caption_rewrites_${MODEL_SLUG}.audit.failed.jsonl"
        ;;
    flickr30k)
        KARPATHY="/workspace/datasets/flickr30k/caption_datasets/dataset_flickr30k.json"
        REWRITES="/workspace/datasets/flickr30k/caption_rewrites_${MODEL_SLUG}.json"
        DUMP_PATH="/workspace/datasets/flickr30k/caption_rewrites_${MODEL_SLUG}.audit.failed.jsonl"
        ;;
    *)
        echo "Unknown dataset: $DATASET (expected coco or flickr30k)"
        exit 1
        ;;
esac

cd /workspace

EXTRA_ARGS=()
if [ "$DUMP_FAILED" = "1" ]; then
    EXTRA_ARGS+=(--dump_failed "$DUMP_PATH")
fi

echo "=========================================="
echo "Dataset:    ${DATASET}"
echo "Model slug: ${MODEL_SLUG}"
echo "Karpathy:   ${KARPATHY}"
echo "Rewrites:   ${REWRITES}"
echo "Dump:       ${DUMP_FAILED} -> ${DUMP_PATH}"
echo "=========================================="

python3 -u tools/check_rewrite_coverage.py \
    --karpathy "$KARPATHY" \
    --rewrites "$REWRITES" \
    "${EXTRA_ARGS[@]}"
