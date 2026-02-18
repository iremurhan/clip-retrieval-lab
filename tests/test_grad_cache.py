"""
Test script to verify gradient correctness of GradCache implementation.

This test compares gradients computed with:
1. Standard training (full batch)
2. Gradient Caching (same batch split into micro-batches)

The gradients should be nearly identical (within floating point tolerance).
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grad_cache import GradCache
from loss import SymmetricInfoNCELoss


class SimpleCLIPModel(nn.Module):
    """
    Minimal CLIP-like model for testing gradient correctness.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(224, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    
    def encode_image(self, images):
        """Encode images to embedding space with L2 normalization."""
        embeds = self.image_encoder(images)
        return nn.functional.normalize(embeds, p=2, dim=-1)
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode text to embedding space with L2 normalization."""
        embeds = self.text_encoder(input_ids)
        return nn.functional.normalize(embeds, p=2, dim=-1)
    
    def forward(self, images, input_ids, attention_mask=None):
        img_embeds = self.encode_image(images)
        txt_embeds = self.encode_text(input_ids, attention_mask)
        return img_embeds, txt_embeds


def test_gradient_correctness():
    """
    Test that GradCache produces the same gradients as standard training.
    """
    print("=" * 70)
    print("Testing Gradient Caching Correctness")
    print("=" * 70)
    
    # Hyperparameters
    batch_size = 16
    micro_batch_size = 4
    embed_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Micro Batch Size: {micro_batch_size}")
    print(f"Embedding Dimension: {embed_dim}\n")
    
    # Create dummy data
    images = torch.randn(batch_size, 224).to(device)
    input_ids = torch.randn(batch_size, 100).to(device)
    attention_mask = torch.ones(batch_size, 100).to(device)
    
    # Configuration for loss
    config = {
        'loss': {
            'temperature': 0.07,
            'intra_img_weight': 0.0,
            'intra_txt_weight': 0.0
        },
        'training': {
            'micro_batch_size': micro_batch_size
        }
    }
    
    # ============================================================
    # TEST 1: Standard Training (Full Batch)
    # ============================================================
    print("Running Test 1: Standard Training (Full Batch)...")
    model_standard = SimpleCLIPModel(embed_dim=embed_dim).to(device)
    criterion = SymmetricInfoNCELoss(config)
    
    # Forward pass
    img_embeds = model_standard.encode_image(images)
    txt_embeds = model_standard.encode_text(input_ids, attention_mask)
    loss_dict = criterion(img_embeds, txt_embeds)
    loss_standard = loss_dict['loss_total']
    
    # Backward pass
    loss_standard.backward()
    
    # Store gradients
    standard_grads = {}
    for name, param in model_standard.named_parameters():
        if param.grad is not None:
            standard_grads[name] = param.grad.clone()
    
    print(f"  Loss: {loss_standard.item():.6f}")
    print(f"  Parameters with gradients: {len(standard_grads)}")
    
    # ============================================================
    # TEST 2: Gradient Caching (Micro Batches)
    # ============================================================
    print("\nRunning Test 2: Gradient Caching (Micro Batches)...")
    
    # Create new model with SAME initialization as standard
    model_cached = SimpleCLIPModel(embed_dim=embed_dim).to(device)
    model_cached.load_state_dict(model_standard.state_dict())
    
    # Create GradCache
    grad_cache = GradCache(
        model=model_cached,
        criterion=criterion,
        config=config,
        device=device,
        scaler=None  # No AMP for this test
    )
    
    # Forward + Backward (handled internally by GradCache)
    loss_dict_cached = grad_cache.forward(images, input_ids, attention_mask)
    loss_cached = loss_dict_cached['loss_total']
    
    # Store gradients
    cached_grads = {}
    for name, param in model_cached.named_parameters():
        if param.grad is not None:
            cached_grads[name] = param.grad.clone()
    
    print(f"  Loss: {loss_cached.item():.6f}")
    print(f"  Parameters with gradients: {len(cached_grads)}")
    
    # ============================================================
    # COMPARISON
    # ============================================================
    print("\n" + "=" * 70)
    print("Gradient Comparison Results")
    print("=" * 70)
    
    # Compare losses
    loss_diff = abs(loss_standard.item() - loss_cached.item())
    print(f"\nLoss Difference: {loss_diff:.10f}")
    
    if loss_diff < 1e-5:
        print("✓ Loss matches (within tolerance)")
    else:
        print("✗ Loss mismatch detected!")
    
    # Compare gradients
    max_grad_diff = 0.0
    avg_grad_diff = 0.0
    num_params = 0
    
    print("\nGradient Comparison by Parameter:")
    print("-" * 70)
    
    for name in standard_grads:
        if name in cached_grads:
            grad_diff = torch.abs(standard_grads[name] - cached_grads[name])
            max_diff = grad_diff.max().item()
            avg_diff = grad_diff.mean().item()
            
            max_grad_diff = max(max_grad_diff, max_diff)
            avg_grad_diff += avg_diff
            num_params += 1
            
            status = "✓" if max_diff < 1e-4 else "✗"
            print(f"{status} {name:40s} Max: {max_diff:.10f}  Avg: {avg_diff:.10f}")
        else:
            print(f"✗ {name:40s} MISSING in cached gradients!")
    
    avg_grad_diff /= num_params
    
    print("-" * 70)
    print(f"\nSummary:")
    print(f"  Max Gradient Difference:     {max_grad_diff:.10f}")
    print(f"  Average Gradient Difference: {avg_grad_diff:.10f}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if max_grad_diff < 1e-4:
        print("✓✓✓ TEST PASSED: Gradients match within tolerance! ✓✓✓")
        print("=" * 70)
        return True
    else:
        print("✗✗✗ TEST FAILED: Gradient mismatch detected! ✗✗✗")
        print("=" * 70)
        return False


if __name__ == "__main__":
    try:
        success = test_gradient_correctness()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗✗✗ TEST ERROR: {e} ✗✗✗")
        import traceback
        traceback.print_exc()
        sys.exit(1)
