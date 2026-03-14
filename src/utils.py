"""
src/utils.py
------------
Utility functions for training and model monitoring.
Provides gradient norm computation for training stability checks.
"""

import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io


def setup_seed(seed=42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_grad_norm(model):
    """
    Compute the total L2 gradient norm across all parameters.
    Useful for monitoring training stability and detecting gradient explosions.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def visualize_text_guided_attention(image, attn_probs, caption):
    """
    Visualize text-guided attention map overlay on an image.

    This function creates a matplotlib figure showing the original image with an
    attention heatmap overlay, indicating which image patches were most relevant
    to the text query.

    Args:
        image: PIL Image or numpy array or torch tensor [3, H, W]
        attn_probs: [1, N] or [N] - Attention probabilities for each patch
        caption: str - The text query/caption for visualization

    Returns:
        fig: matplotlib.pyplot Figure object (caller must close with plt.close(fig))
    """
    # Convert image tensor to PIL if needed
    if isinstance(image, torch.Tensor):
        # Assuming normalized image in [3, H, W] format
        # Denormalize using CLIP statistics
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image = (image * std + mean).clamp(0, 1)
        image = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        image = Image.fromarray(image)

    # Handle attention probs shape [1, N] -> [N]
    if isinstance(attn_probs, torch.Tensor):
        attn_probs = attn_probs.cpu().detach().numpy()
    if attn_probs.ndim == 2:
        attn_probs = attn_probs.squeeze(0)

    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention heatmap
    # Reshape attention to 2D grid (assuming square patches)
    # For ViT-L/14-336: 336/14 = 24 patches per side, +1 for [CLS] = 577 total (24*24+1)
    # attn_probs arriving here is already patches-only ([CLS] excluded by encode_image).
    # Do NOT slice [:-1]; all N entries are valid spatial patches.
    num_patches = len(attn_probs)
    num_patches_side = int(np.sqrt(num_patches))
    if num_patches_side * num_patches_side == num_patches:
        attn_grid = attn_probs.reshape(num_patches_side, num_patches_side)
    else:
        raise ValueError(
            f"attn_probs length {num_patches} is not a perfect square. "
            "Expected patches-only tensor with [CLS] already excluded."
        )

    # Upsample attention to match image size
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    # Create attention map with same resolution as image
    from scipy import ndimage
    attn_upsampled = ndimage.zoom(attn_grid, (img_h / attn_grid.shape[0], img_w / attn_grid.shape[1]), order=1)

    # Overlay attention on image
    axes[1].imshow(image, alpha=0.6)
    im = axes[1].imshow(attn_upsampled, cmap='hot', alpha=0.4)
    axes[1].set_title("Attention Heatmap (Text-Guided)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Attention Weight")

    # Add caption as subtitle
    fig.suptitle(f"Caption: {caption[:60]}...", fontsize=10, wrap=True)

    plt.tight_layout()

    return fig
