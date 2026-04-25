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
import numpy as np
import os
import logging
import random
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

    A pool of photometric transforms from which exactly k are randomly
    selected (without replacement) and applied sequentially.

    Args:
        k (int): Number of transforms to sample from the pool per call.
                 Must satisfy 0 <= k <= pool_size.
        color_jitter_strength (float): Multiplier for (brightness, contrast,
                 saturation); hue is scaled by 0.25x. Default 0.4.
        use_grayscale (bool): Include RandomGrayscale in pool. Default True.

    Raises:
        ValueError: If k > pool_size or k < 0 at construction time.
    """

    def __init__(self, k, color_jitter_strength=0.4, use_grayscale=True):
        s = color_jitter_strength
        self._pool = [
            transforms.ColorJitter(
                brightness=s,
                contrast=s,
                saturation=s,
                hue=s * 0.25,  # hue range is narrower
            ),
            transforms.GaussianBlur(
                kernel_size=23,
                sigma=(0.1, 1.0)
            ),
        ]
        if use_grayscale:
            self._pool.append(transforms.RandomGrayscale(p=1.0))

        self._pool_names = [type(t).__name__ for t in self._pool]

        if not isinstance(k, int) or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k}")
        if k > len(self._pool):
            raise ValueError(
                f"k={k} exceeds photometric pool size={len(self._pool)}. "
                f"Available transforms: {self._pool_names}"
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
            f"pool={self._pool_names})"
        )


def build_anchor_transform(image_size, separate_pipelines=False):
    """
    Build the anchor transform for training (used for inter-modal loss).

    When separate_pipelines=True (SLIP-style), the anchor uses a very mild
    crop (0.9, 1.0) with NO photometric augmentation — preserving visual
    semantics for the image-text contrastive objective.

    When separate_pipelines=False (legacy), same mild crop + flip.

    Args:
        image_size (int): Target spatial resolution.
        separate_pipelines (bool): If True, use minimal augmentation for inter-modal.

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        # Spatial — Mild crop preserving semantic content for text alignment
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Normalization — no photometric augmentation on the anchor
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_augmented_transform(image_size, k, aug_crop_scale_min=0.4,
                              color_jitter_strength=0.4, use_grayscale=True):
    """
    Build the Bipartite Augmentation transform for the contrastive view
    (used for intra-modal L_img_img).

    Phase 1 (Spatial):      RandomResizedCrop + RandomHorizontalFlip
    Phase 2 (Photometric):  StochasticPhotometricPool with k-selection
    Phase 3 (Normalization): ToTensor → CLIP normalization

    Args:
        image_size (int): Target spatial resolution.
        k (int): Number of photometric transforms to sample per image.
        aug_crop_scale_min (float): Lower bound of crop scale (default 0.4).
        color_jitter_strength (float): ColorJitter magnitude (default 0.4).
        use_grayscale (bool): Include RandomGrayscale in pool (default True).

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        # Phase 1 (Spatial)
        transforms.RandomResizedCrop(image_size, scale=(aug_crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Phase 2 (Photometric) — k-selection from configurable pool
        StochasticPhotometricPool(k, color_jitter_strength=color_jitter_strength,
                                  use_grayscale=use_grayscale),
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
# Segment-Feature Loader (B5a/B5b/B5c variants only)
# ============================================================================

_SEG_MODE_FILENAME = {
    "spatial":    "spatial_bins.pt",
    "semantic":   "semantic_ids.pt",
    "continuous": "continuous_features.pt",
}


class SegmentFeatureLoader:
    """
    Loads a precomputed per-patch tensor for the B5a / B5b / B5c variants.

    Unlike the deprecated B5_seg loader, there is no runtime resize / majority
    vote / modulo-wrap: all patch assignments are materialized offline by
    scripts/precompute_sam.py --seg_mode {spatial,semantic,continuous} and
    serialized as a single torch dict keyed by filename stem.

    Layout (under <seg_map_dir>):
        spatial_bins.pt        dict[str -> LongTensor (576,)],   vocab = 28
        semantic_ids.pt        dict[str -> LongTensor (576,)],   vocab = 81
        continuous_features.pt dict[str -> FloatTensor (576, F)], F typically 5

    Args:
        seg_map_dir (str): Directory containing the per-mode .pt file.
        image_size (int): Target spatial resolution (used only for shape checks).
        patch_size (int): ViT patch size (14 for ViT-L/14).
        seg_mode (str): One of 'spatial', 'semantic', 'continuous'.
        seg_vocab_size (int | None): Embedding table size for discrete modes;
            required if seg_mode in {'spatial', 'semantic'}, else ignored.
        seg_feature_dim (int | None): Per-patch feature dim for continuous mode;
            required if seg_mode == 'continuous'.
    """

    def __init__(
        self,
        seg_map_dir: str,
        image_size: int,
        patch_size: int,
        seg_mode: str,
        seg_vocab_size: int | None = None,
        seg_feature_dim: int | None = None,
    ):
        if seg_mode not in _SEG_MODE_FILENAME:
            raise ValueError(
                f"Unknown seg_mode {seg_mode!r}. "
                f"Expected one of {list(_SEG_MODE_FILENAME)}."
            )
        if not os.path.isdir(seg_map_dir):
            raise FileNotFoundError(
                f"seg_map_dir does not exist: {seg_map_dir}. "
                "Run scripts/precompute_sam.py --seg_mode {spatial,semantic,continuous} first."
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}."
            )

        self.seg_map_dir = seg_map_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size           # 24 for 336/14
        self.num_patches = self.grid_size * self.grid_size  # 576
        self.seg_mode = seg_mode
        self.is_continuous = (seg_mode == "continuous")

        pt_path = os.path.join(seg_map_dir, _SEG_MODE_FILENAME[seg_mode])
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(
                f"Precomputed segment features not found: {pt_path}. "
                f"Run: python scripts/precompute_sam.py --seg_mode {seg_mode} ..."
            )

        logger.info(f"Loading segment features: {pt_path}")
        raw = torch.load(pt_path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError(
                f"{pt_path} must contain a dict[str, Tensor], got {type(raw).__name__}"
            )
        # Pack into a single contiguous tensor + numpy stem index. The dict
        # and its str keys would otherwise re-COW per DataLoader worker
        # (refcount writes break shared pages); the packed tensor lives in
        # shared memory and the U-array key table is a single C buffer.
        stems = list(raw.keys())
        first = raw[stems[0]]
        packed = torch.empty((len(stems), *first.shape), dtype=first.dtype)
        for i, s in enumerate(stems):
            packed[i] = raw[s]
        del raw
        packed.share_memory_()
        self._packed = packed                                       # [K, 576] or [K, 576, F]
        order = np.argsort(np.asarray(stems, dtype=np.str_))
        self._stems_sorted = np.asarray(stems, dtype=np.str_)[order]
        self._stems_order = order.astype(np.int64)
        logger.info(
            f"Loaded {len(stems)} segment-feature entries (packed, shared) from {pt_path}"
        )

        if self.is_continuous:
            if seg_feature_dim is None or seg_feature_dim < 1:
                raise ValueError(
                    f"seg_feature_dim must be >= 1 for continuous mode, got {seg_feature_dim!r}"
                )
            self.seg_feature_dim = int(seg_feature_dim)
            self.seg_vocab_size = None
        else:
            if seg_vocab_size is None or seg_vocab_size < 1:
                raise ValueError(
                    f"seg_vocab_size must be >= 1 for seg_mode={seg_mode}, got {seg_vocab_size!r}"
                )
            self.seg_vocab_size = int(seg_vocab_size)
            self.seg_feature_dim = None

    def load(self, filename: str) -> torch.Tensor:
        """
        Return the per-patch tensor for one image.

        Returns:
            LongTensor (576,)        for seg_mode in {'spatial','semantic'}
            FloatTensor (576, F)     for seg_mode == 'continuous'
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        pos = int(np.searchsorted(self._stems_sorted, stem))
        if pos >= self._stems_sorted.shape[0] or str(self._stems_sorted[pos]) != stem:
            raise KeyError(
                f"Precomputed segment features missing for '{stem}' "
                f"(seg_mode={self.seg_mode}, dir={self.seg_map_dir}). "
                "Every training image must be covered."
            )
        feat = self._packed[int(self._stems_order[pos])]

        if self.is_continuous:
            if feat.dim() != 2 or feat.size(0) != self.num_patches \
                    or feat.size(1) != self.seg_feature_dim:
                raise ValueError(
                    f"Continuous features for {stem!r} have shape {tuple(feat.shape)}, "
                    f"expected ({self.num_patches}, {self.seg_feature_dim})."
                )
            return feat.float()

        if feat.dim() != 1 or feat.numel() != self.num_patches:
            raise ValueError(
                f"Discrete IDs for {stem!r} have shape {tuple(feat.shape)}, "
                f"expected ({self.num_patches},)."
            )
        feat = feat.long()
        fmin, fmax = int(feat.min().item()), int(feat.max().item())
        if fmin < 0 or fmax >= self.seg_vocab_size:
            raise ValueError(
                f"seg_ids out of bounds for {stem!r}: "
                f"min={fmin}, max={fmax}, vocab_size={self.seg_vocab_size}"
            )
        return feat


# ============================================================================
# Dataset
# ============================================================================

class _ColumnarSampleStore:
    """COW-safe metadata store. All columns are numpy buffers so that worker
    fork does not break COW via Python refcount writes. __getitem__ rebuilds
    a dict per call to preserve the existing dataset.samples[idx] API."""
    __slots__ = ("image_ids", "sentids", "captions", "filepaths", "filenames")

    def __init__(self, image_ids, sentids, captions, filepaths, filenames):
        self.image_ids = image_ids   # int64  [N]
        self.sentids = sentids       # int64  [N]
        self.captions = captions     # U<max> [N]
        self.filepaths = filepaths   # U<max> [N]
        self.filenames = filenames   # U<max> [N]

    def __len__(self):
        return self.image_ids.shape[0]  # [N] -> scalar

    def __getitem__(self, idx):
        return {
            "image_id": int(self.image_ids[idx]),
            "sentid":   int(self.sentids[idx]),
            "caption":  str(self.captions[idx]),
            "filepath": str(self.filepaths[idx]),
            "filename": str(self.filenames[idx]),
        }


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

        # COW-safe columnar staging. List-of-dicts forces Python refcount
        # writes on every access, multiplying dataset RSS by num_workers on
        # fork. Buffers below are converted to numpy at the end of __init__.
        image_ids, sentids, captions, filepaths, filenames = [], [], [], [], []

        # Karpathy JSON parsing logic
        for img in data['images']:
            current_split = img['split']
            if current_split == 'restval' and split == 'train':
                current_split = 'train'

            if current_split != split:
                continue

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

            fp = img.get('filepath', '')
            fn = img.get('filename', '')
            for sent in img['sentences'][:5]:
                image_ids.append(img_id)
                sentids.append(int(sent['sentid']))
                captions.append(sent['raw'])
                filepaths.append(fp)
                filenames.append(fn)

        del data

        self.samples = _ColumnarSampleStore(
            image_ids=np.asarray(image_ids, dtype=np.int64),
            sentids=np.asarray(sentids, dtype=np.int64),
            captions=np.asarray(captions,  dtype=np.str_),
            filepaths=np.asarray(filepaths, dtype=np.str_),
            filenames=np.asarray(filenames, dtype=np.str_),
        )
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
        # STRICT CONFIG: every aug knob below MUST exist in config['augment'].
        aug_cfg = config['augment']
        k = aug_cfg['k_photometric_augs']
        aug_crop_scale_min = aug_cfg['aug_crop_scale_min']
        color_jitter_strength = aug_cfg['color_jitter_strength']
        use_grayscale = aug_cfg['use_grayscale']
        separate_pipelines = aug_cfg['separate_pipelines']

        transform = build_anchor_transform(image_size, separate_pipelines=separate_pipelines)
        transform_aug = build_augmented_transform(
            image_size, k,
            aug_crop_scale_min=aug_crop_scale_min,
            color_jitter_strength=color_jitter_strength,
            use_grayscale=use_grayscale,
        )

        logger.info(
            f"Train transforms built: anchor=(crop+flip, separate={separate_pipelines}), "
            f"augmented=(crop_min={aug_crop_scale_min}, k={k}, "
            f"jitter={color_jitter_strength}, grayscale={use_grayscale})"
        )
    else:
        # Validation/Test: deterministic, no augmentation
        transform = build_eval_transform(image_size)
        transform_aug = transform

    # B5a/B5b/B5c gating: build SegmentFeatureLoader only if model.seg_mode is set.
    # Every other variant pays zero overhead (no I/O, no extra batch keys).
    seg_loader = None
    seg_mode = config.get('model', {}).get('seg_mode')
    if seg_mode is not None:
        seg_map_dir = config['data']['seg_map_dir']
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
        seg_loader = SegmentFeatureLoader(
            seg_map_dir=seg_map_dir,
            image_size=image_size,
            patch_size=patch_size,
            seg_mode=seg_mode,
            seg_vocab_size=config['model'].get('seg_vocab_size'),
            seg_feature_dim=config['model'].get('seg_feature_dim'),
        )
        logger.info(
            f"B5 seg_loader: mode={seg_mode}, dir={seg_map_dir}, image_size={image_size}, "
            f"patch_size={patch_size}, grid={seg_loader.grid_size}x{seg_loader.grid_size}, "
            f"vocab_size={seg_loader.seg_vocab_size}, feature_dim={seg_loader.seg_feature_dim}"
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