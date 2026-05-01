"""
scripts/preprocess/precompute_sam.py
-------------------------------------
SAM (Segment Anything) precomputation for B5 variants.

Stage 1 — SAM mask generation (default mode, --seg_mode not set):
    For each image, runs SamAutomaticMaskGenerator and serializes the aggregated
    pixel-space segmap (int32, H x W) to <output_dir>/<stem>.npz under key "seg".
    0 = background. This stage is slow (one SAM forward per image).

Stage 2 — per-mode post-processing (--seg_mode spatial|semantic|continuous):
    Reads the .npz files from stage 1 and converts them to per-patch tensors
    for the 336/14 ViT-L grid (576 patches). Output is a single torch dict
    serialized to <output_dir>/<filename>.pt, keyed by filename stem:

        spatial       -> spatial_bins.pt        dict[stem -> LongTensor (576,)]
                         28 bins: 0=bg, 1..27 = 3x3 grid x 3 size bins
        semantic      -> semantic_ids.pt        dict[stem -> LongTensor (576,)]
                         81 ids:  0=bg, 1..80 = COCO categories (contiguous
                         1..80 remap of COCO's sparse category_id space).
                         On COCO: IoU-matched against instances_{train,val}2014.
                         On Flickr30k: CLIP zero-shot classification per mask
                         with the 80 COCO category names as prompts, top-1 if
                         prob > 0.3 else background.
        continuous    -> continuous_features.pt dict[stem -> FloatTensor (576, 5)]
                         per-patch 5-dim geometric vector of the patch's
                         majority segment (bg patches -> zero vector).

Stage 2 does NOT run SAM; it only reads npz files, so it is fast.

B5d — SAM encoder feature extraction (--encoder_features):
    Runs only sam.image_encoder and saves dense frozen encoder features for
    multistream fusion. Per-image shards are written under
    <output_dir>/pool{pool_size}_shards/<stem>.pt; after all shards complete,
    --merge_encoder_features writes the single dict consumed by training:
    <output_dir>/sam_encoder_features_pool{pool_size}.pt.

Usage (stage 1 — SLURM array, 8 shards):
    sbatch --array=0-7 ... \
        --wrap "python scripts/precompute_sam.py \
                --dataset coco --image_dir datasets/coco \
                --output_dir datasets/coco/sam_masks \
                --sam_checkpoint checkpoints/sam_vit_b.pth \
                --shard_id \$SLURM_ARRAY_TASK_ID --num_shards 8"

Usage (stage 2 — single process per mode):
    python scripts/preprocess/precompute_sam.py --seg_mode spatial     --output_dir datasets/coco/sam_masks
    python scripts/preprocess/precompute_sam.py --seg_mode semantic    --output_dir datasets/coco/sam_masks \
        --dataset coco --image_dir datasets/coco \
        --coco_annotations datasets/coco/annotations
    python scripts/preprocess/precompute_sam.py --seg_mode continuous  --output_dir datasets/coco/sam_masks

Usage (B5d encoder features):
    python scripts/preprocess/precompute_sam.py --encoder_features \
        --dataset coco --image_dir datasets/coco \
        --output_dir datasets/coco/sam_encoder_features \
        --sam_checkpoint checkpoints/sam_vit_b.pth --pool_size 8
    python scripts/preprocess/precompute_sam.py --merge_encoder_features \
        --output_dir datasets/coco/sam_encoder_features --pool_size 8
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
# Stage 2 helpers: per-patch feature construction from pixel-space segmaps.
# ----------------------------------------------------------------------------

# COCO 80 "stuff + things" categories in canonical order. The remap table
# below converts sparse COCO category_id (1..90 with gaps) to contiguous
# 1..80 used by the B5b embedding table.
COCO80_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
assert len(COCO80_CATEGORIES) == 80


def _resize_nn(seg_map: np.ndarray, image_size: int) -> np.ndarray:
    """Nearest-neighbor resize a (H, W) int32 segmap to (image_size, image_size)."""
    seg_pil = Image.fromarray(seg_map.astype(np.int32, copy=False), mode="I")
    seg_pil = seg_pil.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    return np.asarray(seg_pil, dtype=np.int64)


def _majority_vote_patches(seg_resized: np.ndarray, grid: int, patch: int) -> np.ndarray:
    """
    Patchify an (image_size, image_size) int64 map into (grid*grid,) by taking
    the most frequent ID inside each (patch, patch) block.
    """
    if seg_resized.shape != (grid * patch, grid * patch):
        raise RuntimeError(
            f"seg_resized has shape {seg_resized.shape}, expected "
            f"{(grid * patch, grid * patch)}."
        )
    max_id = int(seg_resized.max())
    vocab_tmp = max_id + 1
    cells = seg_resized.reshape(grid, patch, grid, patch) \
                       .transpose(0, 2, 1, 3) \
                       .reshape(grid * grid, patch * patch)
    offsets = (np.arange(cells.shape[0], dtype=np.int64) * vocab_tmp)[:, None]
    flat = (cells + offsets).reshape(-1)
    counts = np.bincount(flat, minlength=cells.shape[0] * vocab_tmp)
    counts = counts.reshape(cells.shape[0], vocab_tmp)
    return np.argmax(counts, axis=1).astype(np.int64)  # (num_patches,)


# ---- Spatial (B5a) ---------------------------------------------------------

def assign_spatial_bin(
    mask: np.ndarray,
    image_h: int,
    image_w: int,
    grid_size: int = 3,
    size_bins: int = 3,
) -> int:
    """
    3x3 spatial grid x 3 size bins = 27 foreground bins + 1 background = 28 IDs.

    Returns 0 for empty masks (background); otherwise 1 + grid_bin*3 + size_bin.
    Size thresholds: <5% area -> small, <25% -> medium, else large.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0
    cx, cy = xs.mean() / image_w, ys.mean() / image_h
    area_frac = len(ys) / (image_h * image_w)
    gx = min(int(cx * grid_size), grid_size - 1)
    gy = min(int(cy * grid_size), grid_size - 1)
    grid_bin = gy * grid_size + gx
    if area_frac < 0.05:
        size_bin = 0
    elif area_frac < 0.25:
        size_bin = 1
    else:
        size_bin = 2
    return 1 + grid_bin * size_bins + size_bin


def segmap_to_spatial_patches(seg_map: np.ndarray, image_size: int, patch_size: int) -> torch.Tensor:
    """Produce [576] LongTensor of spatial-bin IDs (0..27) per patch."""
    grid = image_size // patch_size
    H, W = seg_map.shape
    # Per-segment bin lookup, computed at ORIGINAL resolution for accuracy.
    unique_ids = np.unique(seg_map)
    lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.int64)  # seg_id -> bin_id
    for sid in unique_ids:
        if sid == 0:
            lut[0] = 0
            continue
        mask = (seg_map == sid)
        lut[sid] = assign_spatial_bin(mask, H, W)
    # Resize + patchify the raw seg IDs, then apply LUT.
    seg_resized = _resize_nn(seg_map, image_size)
    patch_ids = _majority_vote_patches(seg_resized, grid, patch_size)  # (576,)
    bins = lut[patch_ids]  # (576,)
    if bins.min() < 0 or bins.max() >= 28:
        raise ValueError(f"spatial bin out of range: min={bins.min()} max={bins.max()}")
    return torch.from_numpy(bins).long()


# ---- Continuous (B5c) ------------------------------------------------------

def segmap_to_continuous_patches(
    seg_map: np.ndarray, image_size: int, patch_size: int
) -> torch.Tensor:
    """Produce [576, 5] FloatTensor of geometric features per patch."""
    grid = image_size // patch_size
    num_patches = grid * grid
    H, W = seg_map.shape
    total = H * W

    # Per-segment 5-dim feature [cx, cy, area_frac, log_aspect, patch_frac].
    unique_ids = np.unique(seg_map)
    lut = np.zeros((int(unique_ids.max()) + 1, 5), dtype=np.float32)

    # Need patch-space presence to compute num_patches_fraction. Cheapest path:
    # resize once, count per-ID patch occupancy.
    seg_resized = _resize_nn(seg_map, image_size)
    patch_ids_full = _majority_vote_patches(seg_resized, grid, patch_size)  # (576,)
    patch_counts = np.bincount(patch_ids_full, minlength=int(unique_ids.max()) + 1)

    for sid in unique_ids:
        if sid == 0:
            continue
        ys, xs = np.where(seg_map == sid)
        if len(ys) == 0:
            continue
        cx = float(xs.mean()) / W
        cy = float(ys.mean()) / H
        area_frac = float(len(ys)) / total
        bw = float(xs.max() - xs.min() + 1)
        bh = float(ys.max() - ys.min() + 1)
        ar = bw / max(1.0, bh)
        ar = float(np.clip(ar, 0.2, 5.0))
        log_aspect = float(np.log(ar))
        n_patches = int(patch_counts[sid]) if sid < patch_counts.shape[0] else 0
        patch_frac = n_patches / num_patches
        lut[sid] = [cx, cy, area_frac, log_aspect, patch_frac]

    feats = lut[patch_ids_full]  # (576, 5) fp32, bg rows are zero
    if not np.isfinite(feats).all():
        raise ValueError("continuous features contain NaN/Inf")
    return torch.from_numpy(feats).float()


# ---- Semantic (B5b) --------------------------------------------------------

def _coco_id_remap() -> dict:
    """
    Map COCO's sparse category_id (1..90 with gaps) to contiguous 1..80.
    Keys are COCO JSON category ids; values are our 1..80 embedding indices
    (0 is reserved for background).
    """
    # Standard COCO 1..90 -> 1..80 remap, matching the COCO80_CATEGORIES order.
    coco_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90,
    ]
    assert len(coco_ids) == 80
    return {cid: i + 1 for i, cid in enumerate(coco_ids)}


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two bool/0-1 arrays of identical shape."""
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def _decode_coco_ann(seg_raw, h: int, w: int) -> np.ndarray:
    """Decode a single COCO annotation's `segmentation` field to a bool HxW mask.

    Supports the three standard COCO forms: list-of-polygons, compressed RLE
    (counts=bytes), and uncompressed RLE (counts=list). Called once per SAM
    segment during IoU matching — decoding is lazy to cap peak memory.
    """
    from pycocotools import mask as maskUtils
    if isinstance(seg_raw, list):
        rles = maskUtils.frPyObjects(seg_raw, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg_raw, dict):
        rle = seg_raw if isinstance(seg_raw["counts"], bytes) else maskUtils.frPyObjects([seg_raw], h, w)[0]
    else:
        raise TypeError(f"Unknown COCO segmentation type: {type(seg_raw)}")
    return maskUtils.decode(rle).astype(bool)


def segmap_to_semantic_coco(
    seg_map: np.ndarray,
    image_size: int,
    patch_size: int,
    coco_instance_anns: list,   # list of (cat_id_remapped, h, w, seg_raw) tuples
    iou_threshold: float,
) -> torch.Tensor:
    """Produce [576] LongTensor of COCO-80 category IDs (0=bg, 1..80).

    `coco_instance_anns` holds RAW segmentation fields (polygon / RLE); masks
    are decoded on demand via `_decode_coco_ann` so peak memory is one decoded
    bool mask, not the full set (which would OOM at ~600k anns on COCO train).
    """
    grid = image_size // patch_size
    unique_ids = np.unique(seg_map)
    lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.int64)

    # Decode each instance mask exactly once per image and cache for the
    # inner SAM-segment loop below (seg_map typically has tens of SAM ids, so
    # decoding N_inst times per image would be O(N_sam * N_inst) decodes).
    decoded: list[tuple[int, np.ndarray]] = []
    for cat_id, h, w, seg_raw in coco_instance_anns:
        try:
            inst_mask = _decode_coco_ann(seg_raw, h, w)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to decode ann for cat={cat_id}: {e}")
            continue
        decoded.append((cat_id, inst_mask))

    for sid in unique_ids:
        if sid == 0:
            continue
        sam_mask = (seg_map == sid)
        best_iou = 0.0
        best_cat = 0
        for cat_id, inst_mask in decoded:
            iou = _mask_iou(sam_mask, inst_mask)
            if iou > best_iou:
                best_iou = iou
                best_cat = cat_id
        lut[sid] = best_cat if best_iou >= iou_threshold else 0

    seg_resized = _resize_nn(seg_map, image_size)
    patch_ids = _majority_vote_patches(seg_resized, grid, patch_size)
    cats = lut[patch_ids]
    if cats.min() < 0 or cats.max() >= 81:
        raise ValueError(f"semantic cat out of range: min={cats.min()} max={cats.max()}")
    return torch.from_numpy(cats).long()


def segmap_to_semantic_flickr(
    seg_map: np.ndarray,
    image_pil: "Image.Image",
    image_size: int,
    patch_size: int,
    clip_model,
    clip_processor,
    text_features: "torch.Tensor",
    prob_threshold: float,
    device: str,
) -> torch.Tensor:
    """
    Flickr30K fallback: CLIP zero-shot classify each SAM mask crop against the
    80 COCO category prompts. text_features is [80, D] L2-normalized.
    """
    import torch as _torch
    grid = image_size // patch_size
    unique_ids = np.unique(seg_map)
    lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.int64)

    for sid in unique_ids:
        if sid == 0:
            continue
        mask = (seg_map == sid)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        crop_h = y2 - y1 + 1  # [H, W] bbox height
        crop_w = x2 - x1 + 1  # [H, W] bbox width
        # B5b design decision: discard SAM segments whose bbox is <4 px on
        # either side. Two reasons, both load-bearing:
        #   (1) Correctness — CLIPImageProcessor auto-infers the channel axis;
        #       for shapes like (1, N, 3) it cannot disambiguate (C=1,H=N,W=3)
        #       from (H=1,W=N,C=3) and falls back to channels-first, then
        #       raises "mean must have 1 elements, got 3".
        #   (2) Semantics — CLIP is trained on 224/336-px images. A 1x1 or 3x2
        #       crop upscaled to 336x336 is a blurred color patch with no
        #       object-level signal; its argmax over 80 classes is noise.
        # Skipped segments stay at lut[sid]=0 (background) — identical outcome
        # to a below-threshold zero-shot confidence.
        if crop_h < 4 or crop_w < 4:
            continue
        crop = image_pil.crop((x1, y1, x2 + 1, y2 + 1))
        inputs = clip_processor(images=crop, return_tensors="pt").to(device)
        with _torch.no_grad():
            img_feat = clip_model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            logits = (img_feat @ text_features.T).squeeze(0)  # [80]
            probs = logits.softmax(dim=0)
            top_prob, top_idx = probs.max(dim=0)
        if float(top_prob.item()) >= prob_threshold:
            lut[sid] = int(top_idx.item()) + 1  # 1..80

    seg_resized = _resize_nn(seg_map, image_size)
    patch_ids = _majority_vote_patches(seg_resized, grid, patch_size)
    cats = lut[patch_ids]
    if cats.min() < 0 or cats.max() >= 81:
        raise ValueError(f"semantic cat out of range: min={cats.min()} max={cats.max()}")
    return torch.from_numpy(cats).long()


def _load_coco_instance_annotations(ann_dir: str) -> dict:
    """
    Parse instances_{train,val}2014.json and return a dict
        image_filename_stem -> list[(cat_id_remapped, H, W, seg_raw)]

    `seg_raw` is the annotation's ORIGINAL segmentation field (polygon list or
    RLE dict) — NOT a decoded bool mask. Decoding is deferred to
    `segmap_to_semantic_coco` / `_decode_coco_ann` so peak RSS stays bounded
    (COCO train has ~600k anns; pre-decoding every bool mask OOMs at ~50 GB).
    """
    import json
    try:
        from pycocotools import mask as maskUtils  # noqa: F401 — import check only
    except ImportError as e:
        raise RuntimeError(
            "B5b semantic on COCO requires pycocotools. `pip install pycocotools`."
        ) from e

    remap = _coco_id_remap()
    per_image: dict = {}

    for split in ("train2014", "val2014"):
        path = os.path.join(ann_dir, f"instances_{split}.json")
        if not os.path.isfile(path):
            logger.warning(f"COCO instance annotations not found: {path} (skipping)")
            continue
        logger.info(f"Loading COCO annotations: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        # Build image_id -> (filename_stem, H, W).
        img_meta = {}
        for img in data["images"]:
            stem = os.path.splitext(img["file_name"])[0]
            img_meta[img["id"]] = (stem, img["height"], img["width"])
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            image_id = ann["image_id"]
            if image_id not in img_meta:
                continue
            stem, h, w = img_meta[image_id]
            cat_id = ann["category_id"]
            cat_remapped = remap.get(cat_id)
            if cat_remapped is None:
                continue
            seg = ann.get("segmentation")
            if seg is None:
                continue
            per_image.setdefault(stem, []).append((cat_remapped, h, w, seg))

    logger.info(f"COCO annotations: {len(per_image)} images with >=1 instance")
    return per_image


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Precompute SAM segment maps/per-patch features (B5a/b/c) or "
            "SAM encoder features (B5d)."
        )
    )
    # Stage-selection flag. If set, runs the post-processing pipeline against
    # existing .npz files in --output_dir and writes a single .pt file.
    p.add_argument("--seg_mode", type=str, default=None,
                   choices=["spatial", "semantic", "continuous"],
                   help="Stage 2: post-process existing .npz files into per-patch tensors.")
    p.add_argument("--encoder_features", action="store_true",
                   help="B5d: extract sam.image_encoder features into per-image shard .pt files.")
    p.add_argument("--merge_encoder_features", action="store_true",
                   help="B5d: merge per-image encoder feature shards into one dict.")

    # Stage 1 requires --dataset / --image_dir / --sam_checkpoint.
    # Stage 2 requires --output_dir always; --dataset / --image_dir / --coco_annotations
    # only for seg_mode=semantic.
    p.add_argument("--dataset", type=str, default=None, choices=["coco", "flickr30k"])
    p.add_argument("--image_dir", type=str, default=None,
                   help="Root image directory (e.g. datasets/coco or datasets/flickr30k).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory containing per-image .npz files (stage 1 output / stage 2 input).")
    p.add_argument("--sam_checkpoint", type=str, default=None,
                   help="Path to the sam_vit_b checkpoint (.pth). Required for stage 1.")
    p.add_argument("--sam_model_type", type=str, default="vit_b",
                   help="SAM model type. Default: vit_b.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device. Default: cuda if available else cpu.")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Stage 1 only: index of this shard (0-based).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Stage 1 only: total number of shards.")
    p.add_argument("--points_per_side", type=int, default=32,
                   help="Stage 1 only: SamAutomaticMaskGenerator.points_per_side.")
    p.add_argument("--pool_size", type=int, default=8,
                   help="B5d only: adaptive-average-pool SAM's 64x64 encoder map to this size.")
    p.add_argument("--max_images", type=int, default=None,
                   help="B5d/debug only: cap images after deterministic enumeration.")

    # Stage 2 knobs.
    p.add_argument("--image_size", type=int, default=336,
                   help="Stage 2 only: ViT input resolution. Default 336.")
    p.add_argument("--patch_size", type=int, default=14,
                   help="Stage 2 only: ViT patch size. Default 14.")
    p.add_argument("--coco_annotations", type=str, default=None,
                   help="Stage 2 semantic only (COCO): dir containing "
                        "instances_{train,val}2014.json.")
    p.add_argument("--clip_model_name", type=str,
                   default="openai/clip-vit-large-patch14-336",
                   help="Stage 2 semantic only (Flickr30k): zero-shot CLIP model.")
    p.add_argument("--semantic_iou_threshold", type=float, default=0.3,
                   help="Stage 2 semantic only: min IoU for COCO-instance match.")
    p.add_argument("--semantic_flickr_threshold", type=float, default=0.3,
                   help="Stage 2 semantic only: min CLIP zero-shot prob for Flickr30k.")

    return p.parse_args()


def _run_stage2(args: argparse.Namespace) -> None:
    """Post-process existing .npz files into a single per-mode .pt dict."""
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(
            f"Stage 2 requires stage 1 output at {args.output_dir!r}. "
            "Run SAM first (omit --seg_mode)."
        )

    out_filename = {
        "spatial":    "spatial_bins.pt",
        "semantic":   "semantic_ids.pt",
        "continuous": "continuous_features.pt",
    }[args.seg_mode]
    out_path = os.path.join(args.output_dir, out_filename)
    logger.info(f"Stage 2: seg_mode={args.seg_mode} -> {out_path}")

    npz_files = sorted(
        os.path.join(args.output_dir, f)
        for f in os.listdir(args.output_dir)
        if f.endswith(".npz")
    )
    if not npz_files:
        raise RuntimeError(f"No .npz files found under {args.output_dir}.")
    logger.info(f"Found {len(npz_files)} .npz files under {args.output_dir}")

    # Per-mode heavyweight setup.
    coco_per_image = None
    clip_model = None
    clip_processor = None
    text_features = None
    flickr_image_dir = None

    if args.seg_mode == "semantic":
        if args.dataset is None:
            raise ValueError("--dataset is required for seg_mode=semantic")
        if args.dataset == "coco":
            if not args.coco_annotations:
                raise ValueError("--coco_annotations is required for COCO semantic.")
            coco_per_image = _load_coco_instance_annotations(args.coco_annotations)
        else:  # flickr30k
            if args.image_dir is None:
                raise ValueError("--image_dir is required for flickr30k semantic "
                                 "(zero-shot CLIP reads the raw image).")
            flickr_image_dir = args.image_dir
            from transformers import CLIPModel, CLIPProcessor
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading zero-shot CLIP: {args.clip_model_name} on {device}")
            clip_model = CLIPModel.from_pretrained(args.clip_model_name,
                                                   use_safetensors=True).to(device).eval()
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
            prompts = [f"a photo of a {c}" for c in COCO80_CATEGORIES]
            tok = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                tfeat = clip_model.get_text_features(**tok)
                text_features = tfeat / tfeat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    table: dict = {}
    n_failed = 0
    for path in tqdm(npz_files, desc=f"stage2:{args.seg_mode}", file=sys.stdout):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with np.load(path) as npz:
                seg_map = npz["seg"].astype(np.int32, copy=False)
            if seg_map.ndim != 2:
                raise ValueError(f"seg has shape {seg_map.shape}, expected 2D.")

            if args.seg_mode == "spatial":
                tensor = segmap_to_spatial_patches(seg_map, args.image_size, args.patch_size)
            elif args.seg_mode == "continuous":
                tensor = segmap_to_continuous_patches(seg_map, args.image_size, args.patch_size)
            else:  # semantic
                if args.dataset == "coco":
                    instances = coco_per_image.get(stem, []) if coco_per_image else []
                    tensor = segmap_to_semantic_coco(
                        seg_map, args.image_size, args.patch_size,
                        instances, args.semantic_iou_threshold,
                    )
                else:  # flickr30k
                    # Locate the source image for cropping.
                    img_path = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        candidate = os.path.join(flickr_image_dir, stem + ext)
                        if os.path.isfile(candidate):
                            img_path = candidate
                            break
                    if img_path is None:
                        raise FileNotFoundError(
                            f"Image for stem {stem!r} not found under {flickr_image_dir}"
                        )
                    pil = Image.open(img_path).convert("RGB")
                    tensor = segmap_to_semantic_flickr(
                        seg_map, pil, args.image_size, args.patch_size,
                        clip_model, clip_processor, text_features,
                        args.semantic_flickr_threshold,
                        args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                    )

            table[stem] = tensor
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            logger.error(f"FAILED on {path}: {type(e).__name__}: {e}")

    logger.info(f"Stage 2: built table with {len(table)} entries (failed={n_failed})")

    # Atomic write: .tmp -> rename.
    tmp_path = out_path + ".tmp"
    torch.save(table, tmp_path)
    os.replace(tmp_path, out_path)
    logger.info(f"Wrote {out_path}")

    if n_failed > 0:
        sys.exit(1)


def _sam_feature_shard_dir(output_dir: str, pool_size: int) -> str:
    return os.path.join(output_dir, f"pool{pool_size}_shards")


def _sam_feature_final_path(output_dir: str, pool_size: int) -> str:
    return os.path.join(output_dir, f"sam_encoder_features_pool{pool_size}.pt")


def _preprocess_image_for_sam_encoder(sam, resize_transform, image_path: str, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = resize_transform.apply_image(np.array(image))
    tensor = torch.as_tensor(image_np, device=device)
    tensor = tensor.permute(2, 0, 1).contiguous()[None].float()
    return sam.preprocess(tensor)


@torch.inference_mode()
def _extract_sam_encoder_feature(sam, resize_transform, image_path: str,
                                 device: str, pool_size: int) -> torch.Tensor:
    sam_input = _preprocess_image_for_sam_encoder(sam, resize_transform, image_path, device)
    feat = sam.image_encoder(sam_input)  # [1, 256, 64, 64] for SAM ViT-B
    if pool_size != feat.shape[-1]:
        feat = F.adaptive_avg_pool2d(feat, (pool_size, pool_size))
    return feat.squeeze(0).detach().cpu().to(torch.float16)


def _merge_sam_encoder_features(args: argparse.Namespace) -> None:
    """Merge B5d per-image SAM encoder feature shards into one training dict."""
    src_dir = _sam_feature_shard_dir(args.output_dir, args.pool_size)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"SAM encoder feature shard directory does not exist: {src_dir}")

    files = sorted(
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.endswith(".pt") and not f.endswith(".tmp")
    )
    if not files:
        raise RuntimeError(f"No per-image .pt files found under {src_dir}")

    table = {}
    for path in tqdm(files, desc=f"merge_encoder:pool{args.pool_size}", file=sys.stdout):
        stem = os.path.splitext(os.path.basename(path))[0]
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tensor.dim() != 3:
            raise ValueError(f"{path} has shape {tuple(tensor.shape)}, expected [C, H, W].")
        table[stem] = tensor

    out_path = _sam_feature_final_path(args.output_dir, args.pool_size)
    tmp_path = out_path + ".tmp"
    torch.save(table, tmp_path)
    os.replace(tmp_path, out_path)
    logger.info(f"Wrote merged SAM encoder features: {out_path} ({len(table)} entries)")


def _run_encoder_features(args: argparse.Namespace) -> None:
    """Extract B5d SAM image-encoder features as restartable per-image shards."""
    if args.dataset is None or args.image_dir is None or args.sam_checkpoint is None:
        raise ValueError(
            "B5d encoder feature extraction requires --dataset, --image_dir, "
            "and --sam_checkpoint."
        )
    if args.pool_size < 1:
        raise ValueError(f"--pool_size must be >= 1, got {args.pool_size}")
    if not os.path.isfile(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SAM checkpoint: {args.sam_checkpoint} (type={args.sam_model_type}) on {device}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    all_paths = enumerate_images(args.dataset, args.image_dir)
    if args.max_images is not None:
        all_paths = all_paths[:int(args.max_images)]
    shard = shard_paths(all_paths, args.shard_id, args.num_shards)

    out_dir = _sam_feature_shard_dir(args.output_dir, args.pool_size)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(
        f"Encoder feature shard {args.shard_id}/{args.num_shards}: "
        f"processing {len(shard)} images -> {out_dir}"
    )

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for img_path in tqdm(shard, desc=f"encoder {args.shard_id}/{args.num_shards}", file=sys.stdout):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.pt")
        if os.path.isfile(out_path):
            n_skipped += 1
            continue
        try:
            feat = _extract_sam_encoder_feature(
                sam, resize_transform, img_path, device, args.pool_size)
            tmp_path = out_path + ".tmp"
            torch.save(feat, tmp_path)
            os.replace(tmp_path, out_path)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            logger.error(f"FAILED on {img_path}: {type(e).__name__}: {e}")

    logger.info(
        f"Encoder feature shard {args.shard_id}/{args.num_shards} done. "
        f"written={n_done}, skipped={n_skipped}, failed={n_failed}"
    )
    if n_failed > 0:
        sys.exit(1)
    if args.num_shards == 1:
        _merge_sam_encoder_features(args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    args = parse_args()

    selected_modes = sum([
        args.seg_mode is not None,
        bool(args.encoder_features),
        bool(args.merge_encoder_features),
    ])
    if selected_modes > 1:
        raise ValueError(
            "Choose only one of --seg_mode, --encoder_features, or --merge_encoder_features."
        )

    # B5d dispatches before mask-generation setup. Merge does not need SAM.
    if args.merge_encoder_features:
        if args.pool_size < 1:
            raise ValueError(f"--pool_size must be >= 1, got {args.pool_size}")
        _merge_sam_encoder_features(args)
        return
    if args.encoder_features:
        _run_encoder_features(args)
        return

    # Stage 2 dispatches before any SAM setup.
    if args.seg_mode is not None:
        _run_stage2(args)
        return

    # Stage 1: original SAM mask generation path.
    if args.dataset is None or args.image_dir is None or args.sam_checkpoint is None:
        raise ValueError(
            "Stage 1 (SAM mask generation) requires --dataset, --image_dir, "
            "and --sam_checkpoint. To run stage 2 post-processing, pass --seg_mode."
        )
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
