#!/bin/bash
# =============================================================================
# Merge paraphrase rewrite shards (--shard{i}of{N}.json) into a single
# canonical caption_rewrites_<model_slug>.json. CPU-only; the python tool
# uses standard library plus json.
#
# Usage:
#   sbatch legacy/paraphrase_generation/merge_rewrite_shards.slurm \
#       <dataset> <model_slug> <num_shards>
#
# Examples:
#   sbatch legacy/paraphrase_generation/merge_rewrite_shards.slurm \
#       coco meta-llama-3-8b-instruct 4
#   sbatch legacy/paraphrase_generation/merge_rewrite_shards.slurm \
#       coco mistral-7b-instruct-v0.2 4
#
# After successful merge produces caption_rewrites_<model_slug>.json (and the
# concatenated .failed.jsonl), the per-shard intermediates can be deleted —
# verify the merged file size first.
# =============================================================================
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
# 48G matches the cluster pattern used by build_multilabel.slurm and friends.
# Pyxis squashfs extraction at container start needs ~16G+ RSS; lower limits
# trigger OOM-kill (exit 137) before our python merge code even runs.
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_merge.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_merge.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch <script> <coco|flickr30k> <model_slug> <num_shards>}"
MODEL_SLUG="${2:?Usage: sbatch <script> <dataset> <model_slug> <num_shards>}"
NUM_SHARDS="${3:?Usage: sbatch <script> <dataset> <model_slug> <num_shards>}"

case "$DATASET" in
    coco|flickr30k) ;;
    *) echo "Unknown dataset: $DATASET (expected coco or flickr30k)"; exit 1 ;;
esac

OUTPUT_PATH="/workspace/datasets/${DATASET}/caption_rewrites_${MODEL_SLUG}.json"

# Pre-flight: verify all shard files exist before the merge tool opens any of them.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard="/workspace/datasets/${DATASET}/caption_rewrites_${MODEL_SLUG}.shard${i}of${NUM_SHARDS}.json"
    if [ ! -f "$shard" ]; then
        echo "ERROR: missing shard file: $shard"
        echo "Aborting before partial merge."
        exit 1
    fi
done

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET}"
echo "Model slug:  ${MODEL_SLUG}"
echo "Num shards:  ${NUM_SHARDS}"
echo "Output:      ${OUTPUT_PATH}"
echo "=========================================="

cd /workspace
python3 -u tools/merge_rewrite_shards.py \
    --output "$OUTPUT_PATH" \
    --num_shards "$NUM_SHARDS"

echo ""
echo "=========================================="
echo "Merge complete. Verify the merged JSON sentid count:"
echo "  python3 -c \"import json; print(len(json.load(open('${OUTPUT_PATH}'))))\""
echo "Then reclaim disk:"
echo "  rm /users/beyza.urhan/experiments/datasets/${DATASET}/caption_rewrites_${MODEL_SLUG}.shard*of${NUM_SHARDS}.json"
echo "  rm /users/beyza.urhan/experiments/datasets/${DATASET}/caption_rewrites_${MODEL_SLUG}.shard*of${NUM_SHARDS}.failed.jsonl"
echo "=========================================="
