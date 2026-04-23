#!/bin/bash
# =============================================================================
# B5a / B5b / B5c — Stage 2 precompute: convert existing pixel-space SAM segmaps
# (.npz files produced by scripts/precompute_sam.slurm) into per-patch tensors
# consumed by src/data.py SegmentFeatureLoader.
#
# Stage 1 (`precompute_sam.slurm`) is NOT re-run — the existing .npz masks are
# reused. Stage 2 only reads them.
#
# Usage:
#   sbatch scripts/precompute_sam_stage2.slurm <dataset> <seg_mode>
#
# Examples:
#   sbatch scripts/precompute_sam_stage2.slurm coco      spatial
#   sbatch scripts/precompute_sam_stage2.slurm coco      semantic
#   sbatch scripts/precompute_sam_stage2.slurm coco      continuous
#   sbatch scripts/precompute_sam_stage2.slurm flickr30k spatial
#   sbatch scripts/precompute_sam_stage2.slurm flickr30k semantic
#   sbatch scripts/precompute_sam_stage2.slurm flickr30k continuous
#
# Output (single file per mode, written to <dataset>/sam_masks/):
#   spatial    -> spatial_bins.pt        (LongTensor [576] per image, vocab=28)
#   semantic   -> semantic_ids.pt        (LongTensor [576] per image, vocab=81)
#   continuous -> continuous_features.pt (FloatTensor [576, 5] per image)
#
# Expected wall time on one A100:
#   spatial    / continuous                 : 10–30 min (numpy-only)
#   semantic on coco                        : 2–8 h (IoU against instance masks)
#   semantic on flickr30k                   : 4–12 h (CLIP zero-shot per mask)
# =============================================================================
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:b5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results
#SBATCH --output=/users/beyza.urhan/experiments/results/sam_stage2_%A.out
#SBATCH --error=/users/beyza.urhan/experiments/results/sam_stage2_%A.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch scripts/precompute_sam_stage2.slurm <coco|flickr30k> <spatial|semantic|continuous>}"
SEG_MODE="${2:?Usage: sbatch scripts/precompute_sam_stage2.slurm <coco|flickr30k> <spatial|semantic|continuous>}"

case "$DATASET" in
    coco)
        IMAGE_DIR="/workspace/datasets/coco"
        OUTPUT_DIR="/workspace/datasets/coco/sam_masks"
        COCO_ANN="/workspace/datasets/coco/annotations"
        ;;
    flickr30k)
        IMAGE_DIR="/workspace/datasets/flickr30k/flickr30k_images"
        OUTPUT_DIR="/workspace/datasets/flickr30k/sam_masks"
        COCO_ANN=""
        ;;
    *)
        echo "Unknown dataset: $DATASET (expected coco or flickr30k)"
        exit 1
        ;;
esac

case "$SEG_MODE" in
    spatial|semantic|continuous) ;;
    *) echo "Unknown seg_mode: $SEG_MODE (expected spatial|semantic|continuous)"; exit 1 ;;
esac

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "FATAL: expected stage-1 .npz masks under $OUTPUT_DIR."
    echo "Run: sbatch --array=0-7%4 scripts/precompute_sam.slurm $DATASET"
    exit 1
fi

N_NPZ="$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.npz' | wc -l)"
if [ "$N_NPZ" -eq 0 ]; then
    echo "FATAL: no .npz files under $OUTPUT_DIR. Run stage 1 first."
    exit 1
fi

echo "=========================================="
echo "Dataset:    ${DATASET}"
echo "seg_mode:   ${SEG_MODE}"
echo "Image dir:  ${IMAGE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "# .npz in:  ${N_NPZ}"
[ -n "$COCO_ANN" ] && echo "COCO ann:   ${COCO_ANN}"
echo "=========================================="

cd /workspace

# --- Runtime pip install into a tmpfs-backed dir ---
# The b5 container image does not yet include pycocotools. It ships a native
# _mask.cpython-*.so which must be loadable via mmap(PROT_EXEC). The bind-
# mounted /workspace does NOT satisfy that (fails with "failed to map segment
# from shared object"), so we install into /tmp which is the container's own
# tmpfs and supports executable mapping. /tmp is per-job so we always reinstall.
# --no-deps avoids pulling in numpy 2.x / matplotlib / pillow, which would
# shadow the container's numpy 1.26.4.
PIP_TARGET="/tmp/py_deps"

if [ "$SEG_MODE" = "semantic" ] && [ "$DATASET" = "coco" ]; then
    rm -rf "$PIP_TARGET"
    mkdir -p "$PIP_TARGET"
    echo "Installing pycocotools (no-deps) into ${PIP_TARGET} ..."
    pip install --no-cache-dir --no-deps --target="$PIP_TARGET" "pycocotools==2.0.7"
    export PYTHONPATH="${PIP_TARGET}:/workspace:${PYTHONPATH:-}"
else
    # Other seg_modes / datasets do NOT need pycocotools.
    export PYTHONPATH="/workspace:${PYTHONPATH:-}"
fi

EXTRA_ARGS=()
if [ "$SEG_MODE" = "semantic" ]; then
    EXTRA_ARGS+=(--dataset "$DATASET" --image_dir "$IMAGE_DIR")
    if [ "$DATASET" = "coco" ]; then
        EXTRA_ARGS+=(--coco_annotations "$COCO_ANN")
    fi
fi

python3 -u scripts/precompute_sam.py \
    --seg_mode "$SEG_MODE" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
