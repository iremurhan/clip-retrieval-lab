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
        caption_rewrites_path=None,
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
            caption_rewrites_path (str, optional): Path to LLM-generated caption
                rewrites JSON ({sentid: [r1, r2, r3]}). When set, __getitem__
                uniformly samples from [original, r1, r2, r3] (LaCLIP-style).
        """
        self.images_root_path = images_root_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.transform = transform
        self.transform_aug = transform_aug

        # LaCLIP-style caption augmentation: load precomputed LLM rewrites
        self.caption_rewrites = None
        if caption_rewrites_path:
            if not os.path.exists(caption_rewrites_path):
                raise FileNotFoundError(
                    f"caption_rewrites_path provided but file not found: "
                    f"{caption_rewrites_path}. Run scripts/generate_rewrites.py first."
                )
            import json as _json
            with open(caption_rewrites_path, 'r') as f:
                raw = _json.load(f)
            self.caption_rewrites = {int(k): v for k, v in raw.items()}
            logger.info(
                f"Loaded {len(self.caption_rewrites)} LLM caption rewrites "
                f"from {caption_rewrites_path}"
            )

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

        # Validate coverage: every sample sentid must exist in caption_rewrites
        if self.caption_rewrites is not None:
            dataset_sids = {s['sentid'] for s in self.samples}
            missing = dataset_sids - self.caption_rewrites.keys()
            if missing:
                sample_ids = sorted(missing)[:10]
                raise RuntimeError(
                    f"caption_rewrites is incomplete: {len(missing)}/{len(dataset_sids)} "
                    f"sentids missing (first 10: {sample_ids}). "
                    f"Re-run: sbatch scripts/generate_rewrites.slurm --resume"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']

        # LaCLIP-style: uniform random from [original, rewrite1, ..., rewriteN]
        if self.caption_rewrites is not None:
            sentid = sample['sentid']
            if sentid not in self.caption_rewrites:
                raise KeyError(
                    f"sentid {sentid} not found in caption_rewrites. "
                    f"Rewrites file is incomplete — re-run scripts/generate_rewrites.py."
                )
            candidates = [caption] + self.caption_rewrites[sentid]
            caption = random.choice(candidates)

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

        return {
            'image': img_tensor,
            'image_aug': img_aug_tensor,
            'caption': caption,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'image_id': image_id,
            'sentid': sample['sentid'],
        }


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

    # LaCLIP-style caption rewrites: only for train split
    caption_rewrites_path = None
    if split == 'train':
        para_type = config['paraphraser']['type']
        if para_type == 'llm_precomputed':
            caption_rewrites_path = config['paraphraser']['precomputed_path']

    dataset = CaptionImageDataset(
        images_root_path=images_root,
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],  # defined in config_base.yaml
        split=split,
        transform=transform,
        transform_aug=transform_aug,
        caption_rewrites_path=caption_rewrites_path,
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