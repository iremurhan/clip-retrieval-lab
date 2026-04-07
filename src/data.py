"""
src/data.py
-----------
Data loading and preprocessing for Cross-Modal Retrieval.
Handles dataset loading for Flickr30k and COCO.

Implements a Bipartite Augmentation Architecture for intra-modal contrastive views:
    Phase 1 (Spatial):      Deterministic foundation - RandomResizedCrop + RandomHorizontalFlip
    Phase 2 (Photometric):  Stochastic k-selection pool with VLM-safe magnitudes
    Phase 3 (Normalization): ToTensor → CLIP exact normalization stats

Design Rationale:
    Unlike SimCLR, CLIP-based retrieval maps images to text. Aggressive spatial crops
    (scale < 0.4) or extreme photometric distortions destroy the visual semantics needed
    to match the text prompt, creating false negatives in the contrastive objective.
    All augmentation magnitudes are therefore bounded to VLM-safe ranges.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Configure logger
logger = logging.getLogger(__name__)

# ============================================================================
# CLIP Normalization Constants (from OpenAI CLIP)
# ============================================================================
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ============================================================================
# Bipartite Augmentation Architecture
# ============================================================================

class StochasticPhotometricPool:
    """
    Phase 2 of the Bipartite Augmentation Pipeline.

    A pool of VLM-safe photometric transforms from which exactly k are
    randomly selected (without replacement) and applied sequentially.

    All magnitudes are hardcoded to VLM-safe bounds:
        - ColorJitter: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        - GaussianBlur: sigma=(0.1, 1.0)
        - Grayscale: p=1.0 (applied only when selected)

    Args:
        k (int): Number of transforms to sample from the pool per call.
                 Must satisfy 0 <= k <= pool_size (currently 3).

    Raises:
        ValueError: If k > pool_size or k < 0 at construction time.
    """

    def __init__(self, k):
        # ---- VLM-Safe Hardcoded Magnitudes (DO NOT use SimCLR defaults) ----
        self._pool = [
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.GaussianBlur(
                kernel_size=23,
                sigma=(0.1, 1.0)
            ),
            transforms.RandomGrayscale(p=1.0),
        ]

        if not isinstance(k, int) or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k}")
        if k > len(self._pool):
            raise ValueError(
                f"k={k} exceeds photometric pool size={len(self._pool)}. "
                f"Available transforms: {[type(t).__name__ for t in self._pool]}"
            )
        self._k = k

    def __call__(self, img):
        """
        Apply k randomly selected photometric transforms to img.

        Args:
            img (PIL.Image): Input image (post-spatial, pre-normalization).

        Returns:
            PIL.Image: Photometrically augmented image.
        """
        if self._k == 0:
            return img
        selected = random.sample(self._pool, self._k)
        for t in selected:
            img = t(img)
        return img

    def __repr__(self):
        return (
            f"StochasticPhotometricPool(k={self._k}, "
            f"pool=[ColorJitter, GaussianBlur, RandomGrayscale])"
        )


def build_anchor_transform(image_size):
    """
    Build the deterministic anchor transform for training.
    Spatial: Mild RandomResizedCrop + flip, then normalization.
    NO photometric augmentation on the anchor.

    Args:
        image_size (int): Target spatial resolution.

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        # Phase 1 (Spatial) — Mild crop preserving semantic content
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Phase 3 (Normalization)
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_augmented_transform(image_size, k):
    """
    Build the Bipartite Augmentation transform for the contrastive view.

    Phase 1 (Spatial):      RandomResizedCrop(scale=(0.4, 1.0)) + RandomHorizontalFlip
    Phase 2 (Photometric):  StochasticPhotometricPool with k-selection
    Phase 3 (Normalization): ToTensor → CLIP normalization

    Args:
        image_size (int): Target spatial resolution.
        k (int): Number of photometric transforms to sample per image.

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        # Phase 1 (Spatial) — VLM-safe lower bound at 0.4 to preserve semantics
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Phase 2 (Photometric) — k-selection from VLM-safe pool
        StochasticPhotometricPool(k),
        # Phase 3 (Normalization) — Exact CLIP stats
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_eval_transform(image_size):
    """
    Build the deterministic evaluation transform.
    Resize → CenterCrop → ToTensor → CLIP Normalize.

    Args:
        image_size (int): Target spatial resolution.

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ============================================================================
# Hard Negative Generation (POS-tag swapping)
# ============================================================================

class HardNegativeGenerator:
    """
    Generates hard negative captions via POS-tag based word swapping.
    Swaps nouns, verbs, and adjectives with random alternatives from
    the same batch to create syntactically valid but semantically wrong captions.
    Deterministic given a fixed seed.
    """

    def __init__(self, seed=42):
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
        self.rng = random.Random(seed)

    def generate(self, captions: list[str]) -> list[str]:
        """
        For each caption, generate one hard negative by swapping
        content words (nouns, verbs, adjectives) with words from
        other captions in the batch.

        Returns list of hard negative strings, same length as input.
        Guaranteed to differ from source for captions with swappable words.
        Falls back to word-order shuffle if no swap candidates found.
        """
        docs = list(self.nlp.pipe(captions))
        swap_pos = {"NOUN", "VERB", "ADJ"}

        # Collect all swappable words per POS from the batch
        pool = {}  # pos -> list of words
        for doc in docs:
            for token in doc:
                if token.pos_ in swap_pos and not token.is_stop:
                    pool.setdefault(token.pos_, []).append(token.text)

        hard_negatives = []
        for doc, original in zip(docs, captions):
            tokens = [token.text for token in doc]
            modified = tokens.copy()
            swapped = False

            for i, token in enumerate(doc):
                if token.pos_ in swap_pos and not token.is_stop:
                    candidates = [
                        w for w in pool.get(token.pos_, [])
                        if w.lower() != token.text.lower()
                    ]
                    if candidates and self.rng.random() < 0.5:
                        modified[i] = self.rng.choice(candidates)
                        swapped = True

            if not swapped:
                # Fallback: shuffle word order
                self.rng.shuffle(modified)

            hard_negatives.append(" ".join(modified))

        return hard_negatives


# ============================================================================
# Segment-Map Loader (B5_seg variant only)
# ============================================================================

class SegmentMapLoader:
    """
    Loads precomputed SAM segment maps and converts them to per-patch segment
    IDs for the B5_seg variant. Only instantiated when
    config['model']['seg_embed_size'] is not None.

    Pipeline (per image, deterministic):
        1. Load <seg_map_dir>/<filename_stem>.npz, key "seg" -> int32 (H_orig, W_orig)
        2. Nearest-neighbor resize to (image_size, image_size).
        3. Reshape into (grid, patch, grid, patch) and majority-vote per patch
           -> (grid * grid,) int64 = 576 for ViT-L/14@336.
        4. Modular wrap: seg_id = seg_id % n_seg.

    The loader is intentionally I/O strict: a missing .npz raises
    FileNotFoundError instead of silently substituting zeros.

    Args:
        seg_map_dir (str): Directory containing per-image .npz files.
        image_size (int): Target spatial resolution (e.g. 336).
        patch_size (int): ViT patch size (14 for ViT-L/14).
        n_seg (int): Modular wrap base (segment embedding table size).
    """

    def __init__(self, seg_map_dir: str, image_size: int, patch_size: int, n_seg: int):
        if not os.path.isdir(seg_map_dir):
            raise FileNotFoundError(
                f"seg_map_dir does not exist: {seg_map_dir}. "
                "Run scripts/precompute_sam.py before training the B5_seg variant."
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}."
            )
        if n_seg < 1:
            raise ValueError(f"n_seg must be >= 1, got {n_seg}")

        self.seg_map_dir = seg_map_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 24 for 336/14
        self.num_patches = self.grid_size * self.grid_size  # 576
        self.n_seg = n_seg

    def _resolve_path(self, filename: str) -> str:
        # Key by filename stem so the loader works for both COCO
        # ("COCO_train2014_000000000009.jpg") and Flickr30k ("1000092795.jpg").
        stem = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self.seg_map_dir, f"{stem}.npz")

    def load(self, filename: str) -> torch.Tensor:
        """
        Load and patchify the segment map for one image.

        Returns:
            seg_ids: LongTensor of shape (num_patches,) = (576,) for ViT-L/14@336.
        """
        path = self._resolve_path(filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Segment map not found: {path}. "
                "B5_seg requires precomputed SAM masks for every training image."
            )

        with np.load(path) as npz:
            if "seg" not in npz.files:
                raise KeyError(
                    f"Segment map file {path} missing required key 'seg'. "
                    f"Available keys: {npz.files}"
                )
            seg = npz["seg"]  # (H_orig, W_orig), int32

        if seg.ndim != 2:
            raise ValueError(
                f"Segment map at {path} has shape {seg.shape}, expected 2D (H, W)."
            )

        # Step 1: nearest-neighbor resize to (image_size, image_size).
        # PIL.Image.resize with NEAREST never blends IDs across boundaries.
        # We cast to int32 first because PIL only accepts int32 / uint8 / float
        # for mode='I'; int32 round-trips losslessly through mode='I'.
        seg_int32 = seg.astype(np.int32, copy=False)
        seg_pil = Image.fromarray(seg_int32, mode="I")
        seg_pil = seg_pil.resize(
            (self.image_size, self.image_size), resample=Image.Resampling.NEAREST
        )
        seg_resized = np.asarray(seg_pil, dtype=np.int64)  # (image_size, image_size)

        if seg_resized.shape != (self.image_size, self.image_size):
            raise RuntimeError(
                f"Resized segment map has shape {seg_resized.shape}, "
                f"expected ({self.image_size}, {self.image_size})."
            )

        # Step 2: Apply modular wrap BEFORE majority vote so patches that span
        # multiple raw IDs (which would all be wrapped to the same bucket) cast
        # their votes consistently.
        seg_resized = seg_resized % self.n_seg

        # Step 3: Reshape into (grid, patch_size, grid, patch_size) and
        # majority-vote per (grid, grid) cell.
        g, p = self.grid_size, self.patch_size
        # (g, p, g, p) -> (g, g, p, p) -> (g*g, p*p)
        cells = seg_resized.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(g * g, p * p)

        # np.bincount along axis=1 with minlength=n_seg, then argmax.
        # Vectorized: build offsets so each row's bincount is independent.
        # Shape transformation: (576, 196) -> (576,)
        offsets = (np.arange(cells.shape[0], dtype=np.int64) * self.n_seg)[:, None]
        flat = (cells + offsets).reshape(-1)
        counts = np.bincount(flat, minlength=cells.shape[0] * self.n_seg)
        counts = counts.reshape(cells.shape[0], self.n_seg)  # (576, n_seg)
        majority = np.argmax(counts, axis=1).astype(np.int64)  # (576,)

        seg_ids = torch.from_numpy(majority).long()  # (576,)
        # Bounds check: every ID must lie within [0, n_seg).
        if seg_ids.min().item() < 0 or seg_ids.max().item() >= self.n_seg:
            raise ValueError(
                f"seg_ids out of bounds at {path}: "
                f"min={seg_ids.min().item()}, max={seg_ids.max().item()}, n_seg={self.n_seg}"
            )
        return seg_ids


# ============================================================================
# Dataset
# ============================================================================

class CaptionImageDataset(Dataset):
    def __init__(
        self,
        images_root_path,
        captions_path,
        tokenizer,
        max_length=77,
        split='train',
        transform=None,
        transform_aug=None,
        seg_loader=None,
    ):
        """
        Args:
            images_root_path (str): Path to image folder.
            captions_path (str): Path to Karpathy JSON file.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Token sequence length (Default: 77).
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Anchor image transform.
            transform_aug (callable, optional): Augmented view transform (intra-modal).
        """
        self.images_root_path = images_root_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.transform = transform
        self.transform_aug = transform_aug
        # B5_seg only: per-image SAM segment-map loader. None for all other variants.
        self.seg_loader = seg_loader

        # Load Captions (Karpathy JSON)
        logger.info(f"Loading captions from {captions_path} for split: {split}")
        with open(captions_path, 'r') as f:
            data = json.load(f)

        self.samples = []

        # Karpathy JSON parsing logic
        for img in data['images']:
            current_split = img['split']
            if current_split == 'restval' and split == 'train':
                current_split = 'train'

            if current_split == split:
                # Handle ID variations (COCO: cocoid/id, Flickr30k: imgid)
                if 'cocoid' in img:
                    img_id = int(img['cocoid'])
                elif 'imgid' in img:
                    img_id = int(img['imgid'])
                elif 'id' in img:
                    img_id = int(img['id'])
                else:
                    raise ValueError(
                        f"Image entry missing 'cocoid', 'imgid', or 'id'. Keys: {list(img.keys())}"
                    )

                # Take exactly 5 captions
                sentences = img['sentences'][:5]
                for sent in sentences:
                    self.samples.append({
                        'image_id': img_id,
                        'caption': sent['raw'],
                        'sentid': int(sent['sentid']),
                        'filepath': img.get('filepath', ''),
                        'filename': img.get('filename', '')
                    })

        logger.info(f"Found {len(self.samples)} samples for split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']

        # Load Image
        filepath = sample.get('filepath', '').strip()
        filename = sample['filename']

        if filepath:
            image_path = os.path.join(self.images_root_path, filepath, filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(self.images_root_path, filename)
        else:
            image_path = os.path.join(self.images_root_path, filename)

        # Fail fast on missing images
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        # Transform Image (original anchor and augmented contrastive view)
        img_tensor = self.transform(image)
        img_aug_tensor = self.transform_aug(image)

        # Tokenize Text
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        out = {
            'image': img_tensor,
            'image_aug': img_aug_tensor,
            'caption': caption,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'image_id': image_id,
            'sentid': sample['sentid'],
        }

        # B5_seg only: attach per-patch segment IDs alongside the existing
        # batch entries. Strictly gated by the presence of seg_loader so that
        # other variants pay zero I/O / collate cost.
        if self.seg_loader is not None:
            out['seg_ids'] = self.seg_loader.load(filename)  # LongTensor (576,)

        return out


# ============================================================================
# DataLoader Factory
# ============================================================================

def create_image_text_dataloader(config, tokenizer, split='train'):
    """
    Factory function to create DataLoader for image-text retrieval datasets.

    Creates a DataLoader for Flickr30k or COCO datasets with appropriate
    transforms, shuffling, and batch size based on the split.

    STRICT CONFIG:
        - config['data']['images_path']
        - config['data']['captions_path']
        - config['training']['batch_size']
        - config['augment']['k_photometric_augs']  (REQUIRED, raises KeyError if missing)
    """
    shuffle = (split == 'train')
    images_root = config['data']['images_path']
    captions_path = config['data']['captions_path']

    # Resolve image size from data config (defined in config_base.yaml)
    image_size = config['data']['image_size']

    if split == 'train':
        # STRICT CONFIG: k_photometric_augs MUST exist. No fallback.
        k = config['augment']['k_photometric_augs']

        transform = build_anchor_transform(image_size)
        transform_aug = build_augmented_transform(image_size, k)

        logger.info(
            f"Train transforms built: anchor=(crop+flip), "
            f"augmented=(bipartite, k={k} photometric from pool)"
        )
    else:
        # Validation/Test: deterministic, no augmentation
        transform = build_eval_transform(image_size)
        transform_aug = transform

    # B5_seg gating: build SegmentMapLoader only if model.seg_embed_size is set.
    # Every other variant pays zero overhead (no .npz I/O, no extra batch keys).
    seg_loader = None
    if config.get('model', {}).get('seg_embed_size') is not None:
        seg_map_dir = config['data']['seg_map_dir']
        n_seg = config['model']['seg_embed_size']
        # patch_size is derived from the CLIP model name, not hardcoded:
        # 'clip-vit-large-patch14-336' -> patch=14. Matches the only model
        # supported by this project (see configs/config_base.yaml).
        model_name = config['model']['image_model_name']
        if 'patch14' in model_name:
            patch_size = 14
        elif 'patch16' in model_name:
            patch_size = 16
        elif 'patch32' in model_name:
            patch_size = 32
        else:
            raise ValueError(
                f"Cannot infer ViT patch size from image_model_name={model_name!r}. "
                "Add a branch to create_image_text_dataloader for this model."
            )
        seg_loader = SegmentMapLoader(
            seg_map_dir=seg_map_dir,
            image_size=image_size,
            patch_size=patch_size,
            n_seg=n_seg,
        )
        logger.info(
            f"B5_seg seg_loader: dir={seg_map_dir}, image_size={image_size}, "
            f"patch_size={patch_size}, grid={seg_loader.grid_size}x{seg_loader.grid_size}, "
            f"n_seg={n_seg}"
        )

    dataset = CaptionImageDataset(
        images_root_path=images_root,
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],  # defined in config_base.yaml
        split=split,
        transform=transform,
        transform_aug=transform_aug,
        seg_loader=seg_loader,
    )

    seed = config['training']['seed']

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        import numpy as np
        import random
        import torch as _torch
        _torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if split == 'train':
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],  # defined in config_base.yaml
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],  # defined in config_base.yaml
        pin_memory=True,
        drop_last=(split == 'train'),
        worker_init_fn=_worker_init_fn,
        generator=g,
    )

    return loader