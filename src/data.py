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

        return {
            'image': img_tensor,
            'image_aug': img_aug_tensor,
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

    # Resolve image size from model config (336 for CLIP ViT-L/14@336px)
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

    dataset = CaptionImageDataset(
        images_root_path=images_root,
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        split=split,
        transform=transform,
        transform_aug=transform_aug,
    )

    # Debug Truncation
    if config.get('debug', {}).get('debug_mode', False):
        debug_limit = config['debug'].get('debug_samples', 100)
        if len(dataset.samples) > debug_limit:
            logger.warning(f"DEBUG MODE: Truncating dataset to {debug_limit} samples.")
            dataset.samples = dataset.samples[:debug_limit]

    seed = config.get('training', {}).get('seed', 42)

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        import numpy as np
        import random
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train'),
        worker_init_fn=_worker_init_fn,
    )

    return loader