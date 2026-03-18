#!/usr/bin/env python3
"""
scripts/visualize_augs.py
-------------------------
Standalone sanity-check script for the Bipartite Augmentation Pipeline.

Loads a batch from the training dataset, mathematically inverts the exact CLIP
normalization, and plots anchor vs. augmented view side-by-side.

This script is DECOUPLED from the training pipeline — matplotlib is imported
here only, never inside src/data.py.

Usage:
    python scripts/visualize_augs.py --config configs/config_flickr.yaml
    python scripts/visualize_augs.py --config configs/config_coco.yaml --num_samples 8
"""

import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import create_image_text_dataloader, CLIP_MEAN, CLIP_STD
from src.setup import setup_config


def denormalize(tensor, mean, std):
    """
    Mathematically invert CLIP normalization: x_original = x_normalized * std + mean

    Args:
        tensor (Tensor): [C, H, W] normalized image tensor.
        mean (list): Per-channel means used during normalization.
        std (list): Per-channel stds used during normalization.

    Returns:
        ndarray: [H, W, C] uint8 image in [0, 255] range.
    """
    # Clone to avoid mutating original
    img = tensor.clone().cpu()

    # Invert: x = x_norm * std + mean
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    # Clamp to valid range and convert
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
    return (img * 255).astype(np.uint8)


def visualize_batch(batch, num_samples=4):
    """
    Plot anchor and augmented views side-by-side.

    Args:
        batch (dict): Batch from CaptionImageDataset.
        num_samples (int): Number of samples to display.
    """
    images = batch['image']
    images_aug = batch['image_aug']

    num_samples = min(num_samples, images.size(0))

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize with exact CLIP stats
        anchor = denormalize(images[i], CLIP_MEAN, CLIP_STD)
        augmented = denormalize(images_aug[i], CLIP_MEAN, CLIP_STD)

        axes[i, 0].imshow(anchor)
        axes[i, 0].set_title(f"Anchor (sample {i})", fontsize=11)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(augmented)
        axes[i, 1].set_title(f"Augmented View (sample {i})", fontsize=11)
        axes[i, 1].axis('off')

    plt.suptitle(
        "Bipartite Augmentation: Anchor vs. Contrastive View\n"
        "(Phase 1: Spatial → Phase 2: Photometric k-Selection → Phase 3: CLIP Norm)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'augmentation_sanity_check.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {os.path.abspath(output_path)}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Bipartite Augmentation Pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to dataset config YAML")
    parser.add_argument('--num_samples', type=int, default=4, help="Number of samples to visualize")
    args = parser.parse_args()

    # Load config (uses setup_config from src/setup.py)
    config = setup_config(config_path=args.config)

    # Add image_size and data keys if they come from training section
    if 'image_size' not in config.get('data', {}):
        config.setdefault('data', {})['image_size'] = 336
    if 'max_length' not in config.get('data', {}):
        config.setdefault('data', {})['max_length'] = config['training']['max_length']
    if 'num_workers' not in config.get('data', {}):
        config.setdefault('data', {})['num_workers'] = config['training']['num_workers']

    # We need a tokenizer — use the CLIP tokenizer
    from transformers import CLIPTokenizerFast
    model_name = config['model']['image_model_name']
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    # Create train DataLoader (to see augmented views)
    loader = create_image_text_dataloader(config, tokenizer, split='train')

    # Get one batch
    batch = next(iter(loader))

    print(f"Batch size:     {batch['image'].shape[0]}")
    print(f"Image shape:    {batch['image'].shape}")
    print(f"Image aug shape: {batch['image_aug'].shape}")
    print(f"k_photometric:  {config['augment']['k_photometric_augs']}")

    visualize_batch(batch, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
