"""
src/utils.py
------------
Utility functions for training and model monitoring.
Provides gradient norm computation for training stability checks.
"""

import torch


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
