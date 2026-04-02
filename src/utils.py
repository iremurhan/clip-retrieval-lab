"""
src/utils.py
------------
Utility functions for training and model monitoring.
Provides gradient norm computation for training stability checks.
"""

import math
import logging
import torch

logger = logging.getLogger(__name__)


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
    if not math.isfinite(total_norm):
        logger.critical(f"Non-finite grad norm detected: {total_norm}")
    return total_norm


@torch.no_grad()
def chunked_matmul(a, b_t, chunk_size=1024):
    """
    Compute a @ b_t^T in row-chunks to limit peak RAM.

    Args:
        a:  Tensor [N, D]
        b_t: Tensor [M, D]  (will be transposed internally)
        chunk_size: number of rows of `a` per chunk

    Returns:
        sims: Tensor [N, M]  (allocated once, filled in-place)
    """
    n = a.shape[0]
    m = b_t.shape[0]
    sims = torch.empty(n, m, dtype=a.dtype)  # [N, M]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims[start:end] = torch.matmul(a[start:end], b_t.t())  # [chunk, D] x [D, M] -> [chunk, M]
    return sims
