"""
scripts/precompute_sam.py
-------------------------
Offline SAM (Segment Anything) mask precomputation for the B5_seg variant.

For each image in the dataset, runs SamAutomaticMaskGenerator with the
sam_vit_b checkpoint and converts the resulting list of binary masks into a
single integer segment map of shape (H_orig, W_orig):

    - Background pixels (uncovered by any mask) -> 0
    - Foreground pixels -> integer ID >= 1
    - In case of overlap, the pixel is assigned the ID of the mask with the
      highest predicted IoU (predicted_iou returned by SAM).

The segment map is saved as a compressed .npz file under
    <output_dir>/<image_filename_stem>.npz
keyed by the image filename stem (e.g. "COCO_train2014_000000000009" for COCO
or "1000092795" for Flickr30k). Each .npz contains a single key "seg" of dtype
int32 at the original image resolution. No resizing is applied at this stage.

The script supports SLURM-friendly sharding via --shard_id / --num_shards so
multiple jobs can process disjoint slices of the image list concurrently. It is
restartable: any image whose .npz already exists is skipped.

Usage (single process):
    python scripts/precompute_sam.py \
        --dataset coco \
        --image_dir datasets/coco \
        --output_dir precomputed/sam_masks/coco \
        --sam_checkpoint checkpoints/sam_vit_b.pth

Usage (SLURM array, 8 shards):
    sbatch --array=0-7 ... \
        --wrap "python scripts/precompute_sam.py \
                --dataset coco --image_dir datasets/coco \
                --output_dir precomputed/sam_masks/coco \
                --sam_checkpoint checkpoints/sam_vit_b.pth \
                --shard_id \$SLURM_ARRAY_TASK_ID --num_shards 8"
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Image enumeration
# ----------------------------------------------------------------------------

# Recognized image suffixes. SAM is RGB-only, so we accept the standard formats
# present in COCO / Flickr30k.
_IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


def enumerate_images(dataset: str, image_dir: str) -> list[str]:
    """
    Walk the dataset directory and return a sorted list of absolute image
    paths. Sorting is deterministic so that --shard_id always selects the same
    subset across runs.

    For COCO we recurse into train2014/ and val2014/ (matching the on-disk
    layout used by this project). For Flickr30k all images live directly under
    flickr30k_images/.

    Raises:
        FileNotFoundError: if the resolved image directory does not exist.
        RuntimeError: if no images are found.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    if dataset == "coco":
        # COCO Karpathy splits live in train2014/ and val2014/.
        candidate_subdirs = ["train2014", "val2014"]
        roots = [os.path.join(image_dir, sd) for sd in candidate_subdirs]
        roots = [r for r in roots if os.path.isdir(r)]
        if not roots:
            raise FileNotFoundError(
                f"For dataset=coco, expected at least one of "
                f"{candidate_subdirs} under {image_dir}, found none."
            )
    elif dataset == "flickr30k":
        # Flickr30k images live in flickr30k_images/, but we also accept
        # the case where image_dir already points to that folder.
        sub = os.path.join(image_dir, "flickr30k_images")
        roots = [sub] if os.path.isdir(sub) else [image_dir]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected 'coco' or 'flickr30k'.")

    paths: list[str] = []
    for root in roots:
        for entry in os.listdir(root):
            if entry.lower().endswith(_IMG_SUFFIXES):
                paths.append(os.path.join(root, entry))

    if not paths:
        raise RuntimeError(f"No images found under {roots}")

    paths.sort()
    return paths


def shard_paths(paths: list[str], shard_id: int, num_shards: int) -> list[str]:
    """Return the slice of `paths` assigned to this shard (round-robin)."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    return paths[shard_id::num_shards]


# ----------------------------------------------------------------------------
# Mask aggregation
# ----------------------------------------------------------------------------

def masks_to_segmap(masks: list[dict], h: int, w: int) -> np.ndarray:
    """
    Collapse SAM's list of mask dicts into a single (H, W) int32 segment map.

    SAM returns one dict per mask with at least the keys:
        - "segmentation": (H, W) bool ndarray
        - "predicted_iou": float
    Pixel ownership in case of overlap is resolved by predicted_iou: the mask
    with the higher predicted IoU wins. Background (uncovered) stays 0. Segment
    IDs start at 1 and follow the order in which masks were assigned (sorted by
    predicted_iou descending so that the highest-IoU masks get the lowest IDs).

    Args:
        masks: List of dicts as returned by SamAutomaticMaskGenerator.generate().
        h: Original image height.
        w: Original image width.

    Returns:
        seg_map: int32 ndarray of shape (H, W). 0 = background.
    """
    seg_map = np.zeros((h, w), dtype=np.int32)
    if not masks:
        return seg_map

    # Sort masks by predicted_iou DESCENDING. Highest-confidence masks get
    # priority and the lowest IDs. By iterating in this order and only writing
    # to currently-background pixels, the highest-IoU mask wins every overlap.
    sorted_masks = sorted(masks, key=lambda m: float(m["predicted_iou"]), reverse=True)

    next_id = 1
    for m in sorted_masks:
        seg = m["segmentation"]
        if seg.shape != (h, w):
            raise ValueError(
                f"SAM returned mask of shape {seg.shape}, expected ({h}, {w}). "
                "Refusing to silently resize."
            )
        # Only assign to background pixels — preserves higher-IoU masks first.
        write = seg & (seg_map == 0)
        if not write.any():
            continue
        seg_map[write] = next_id
        next_id += 1

    return seg_map


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute SAM segment maps for COCO / Flickr30k."
    )
    p.add_argument("--dataset", type=str, required=True, choices=["coco", "flickr30k"])
    p.add_argument("--image_dir", type=str, required=True,
                   help="Root image directory (e.g. datasets/coco or datasets/flickr30k).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write per-image .npz files.")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to the sam_vit_b checkpoint (.pth).")
    p.add_argument("--sam_model_type", type=str, default="vit_b",
                   help="SAM model type. Default: vit_b.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device. Default: cuda if available else cpu.")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Index of this shard (0-based) for parallelism.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards.")
    p.add_argument("--points_per_side", type=int, default=32,
                   help="SamAutomaticMaskGenerator.points_per_side.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    args = parse_args()

    if not os.path.isfile(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Lazy import: keep top-level imports cheap and fail loudly if SAM is
    # missing only when the script is actually invoked.
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SAM checkpoint: {args.sam_checkpoint} (type={args.sam_model_type}) on {device}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
    )

    all_paths = enumerate_images(args.dataset, args.image_dir)
    logger.info(f"Found {len(all_paths)} images under {args.image_dir}")

    shard = shard_paths(all_paths, args.shard_id, args.num_shards)
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: processing {len(shard)} images "
        f"-> {args.output_dir}"
    )

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for img_path in tqdm(shard, desc=f"shard {args.shard_id}/{args.num_shards}", file=sys.stdout):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}.npz")

        # Restartability: skip if already produced.
        if os.path.isfile(out_path):
            n_skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)  # (H, W, 3) uint8
            h, w = image_np.shape[:2]

            with torch.inference_mode():
                masks = mask_generator.generate(image_np)

            seg_map = masks_to_segmap(masks, h, w)  # (H, W) int32

            # Atomic write: .tmp -> rename. Prevents half-written files on crash.
            # NOTE: np.savez_compressed auto-appends ".npz" when passed a PATH,
            # turning "foo.npz.tmp" into "foo.npz.tmp.npz" and breaking the
            # subsequent os.replace. Passing a file object suppresses that.
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, seg=seg_map)
            os.replace(tmp_path, out_path)

            n_done += 1
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            logger.error(f"FAILED on {img_path}: {type(e).__name__}: {e}")

    logger.info(
        f"Shard {args.shard_id}/{args.num_shards} done. "
        f"written={n_done}, skipped={n_skipped}, failed={n_failed}"
    )

    if n_failed > 0:
        # Fail-loud: non-zero exit so SLURM marks the array task as failed.
        sys.exit(1)


if __name__ == "__main__":
    main()
