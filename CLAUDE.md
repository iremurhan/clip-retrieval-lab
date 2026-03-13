# CLAUDE.md

This file provides strict guidance to Claude Code regarding the `clip-retrieval-lab` project.

## Role & Identity
You are a Senior Machine Learning Researcher and Expert PyTorch Developer. 
- Be concise, direct, and technical. 
- NO conversational filler or emojis.
- ALL code, comments, and commit messages must be in English.

## Project Overview
CLIP-based cross-modal image–text retrieval. Fine-tuning `openai/clip-vit-large-patch14-336` on **Flickr30k** and **MS-COCO** using Symmetric InfoNCE loss with optional False Negative Elimination (FNE).

## Environment & HPC (SLURM) Rules
- **Login Node vs. Compute Node:** NEVER run heavy training or mining directly on the login node.
- **Interactive Testing:** Use `srun` for quick tests or debugging. 
  - *Example:* `srun --gpus=1 --mem=32G --partition=debug python run.py --config configs/config_flickr.yaml --override debug.debug_mode=true`
- **Long-Running Jobs:** Use `sbatch` via the provided wrapper scripts for full training runs.
- **Docker/Singularity:** Code runs within `biremurhan/image-text-contrast:v0.5`.
- **VRAM Management:** Use GradCache for large batches. 
- **WandB:** Credentials located in `~/experiments/env/wandb.env`.

## Research & Coding Principles
1. **Tensor Shapes are Sacred:** Every dimension-changing operation (view, transpose, matmul) MUST have an inline comment. Example: `x = x.squeeze(1) # [B, 1, D] -> [B, D]`
2. **Fail Fast:** Raise `ValueError` or `RuntimeError` for shape/logic mismatches. No silent fallbacks.
3. **Minimalism:** Only modify specific lines required. No unnecessary refactoring.
4. **Dependencies:** NEVER attempt to install `torch` or `torchvision`.

## Common Commands (HPC)

**Training (Batch)**
```bash
# Production runs should use the wrapper
./scripts/start_training.sh <run_name> [config_path]
```

**Debug Mode (Interactive)**

```bash
srun --gpus=1 python run.py --config configs/config_flickr.yaml --override debug.debug_mode=true debug.debug_samples=100
```