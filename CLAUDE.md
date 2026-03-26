# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIP-based cross-modal image–text retrieval. Fine-tunes `openai/clip-vit-large-patch14-336` on **Flickr30k** and **MS-COCO** using Symmetric InfoNCE loss with optional hard-negative mining (False Negative Elimination).

## Environment Setup

```bash
# Activate the local venv (Python 3.13)
source venv/bin/activate

# Install dependencies (torch is NOT in requirements.txt — provide it separately)
pip install -r requirements.txt
```

Docker image for HPC: `biremurhan/image-text-contrast:v0.5` (includes PyTorch).

## Common Commands

**Training (local)**
```bash
python run.py --config configs/config_flickr.yaml
python run.py --config configs/config_coco.yaml

# With config overrides
python run.py --config configs/config_flickr.yaml --override training.epochs=10 loss.temperature=0.07

# Resume from checkpoint
python run.py --config configs/config_flickr.yaml --resume checkpoints/best_model.pth
```

**Training (HPC/SLURM)**
```bash
./scripts/start_training.sh <run_name> [config_path]
```

**Pairwise similarity mining (required before training with hard negatives)**
```bash
# Text-text caption similarity (most common — used for FNE in training)
python tools/mine_pairwise_sim.py --modality caption --config configs/config_flickr.yaml --top_k 1000

# Image-image visual similarity
python tools/mine_pairwise_sim.py --modality visual --config configs/config_flickr.yaml

# Image-image caption consensus (saves mean/min/max variants)
python tools/mine_pairwise_sim.py --modality consensus --config configs/config_flickr.yaml

# HPC wrapper
./scripts/start_mining.sh flickr30k   # or coco
```

**Debug mode (fast iteration, uses train set for val)**
```bash
python run.py --config configs/config_flickr.yaml --override debug.debug_mode=true debug.debug_samples=100
```

## Config System

Configs are hierarchical YAML files merged at runtime:

1. `configs/config_base.yaml` — all defaults (model, loss, training, augment, debug, logging, mining)
2. `configs/config_{coco,flickr}.yaml` — dataset paths and mining paths; deep-merged on top of base
3. `--override key.subkey=value` — CLI overrides applied last

`setup_config()` in `src/setup.py` handles the merge. Dataset name (coco vs flickr) is inferred from `data.images_path`.

WandB **requires** `logging.wandb_project` to be set in the config (or via override) — if missing, WandB is silently skipped.

## Architecture

```
run.py                  Training entry point: config → dataloader → DualEncoder → Trainer.fit()
src/
  setup.py              Config loading (deep-merge + CLI overrides), WandB init, seed setup
  data.py               CaptionImageDataset (Karpathy JSON), FNE hard-negative sampling, DataLoader factory
  model.py              DualEncoder: CLIP backbone + optional projection heads, selective freezing
  loss.py               SymmetricInfoNCELoss: inter-modal InfoNCE + optional intra-modal consistency
  train.py              Trainer: train_epoch, evaluate, checkpoint save/load, WandB logging
  metrics.py            compute_recall_at_k, compute_map_at_k (Recall@1/5/10, MAP@5/10)
  utils.py              compute_grad_norm and misc helpers
tools/
  mine_pairwise_sim.py  Standalone mining script (caption / visual / consensus modalities)
  create_triplets.py    Triplet creation utility
configs/
  config_base.yaml      All defaults
  config_{coco,flickr}.yaml   Dataset-specific paths
  sweep*.yaml           W&B sweep configs
scripts/
  *.slurm               SLURM job scripts
  start_training.sh / start_mining.sh   HPC wrappers
```

### Key Design Decisions

**DualEncoder (`src/model.py`):** Wraps `CLIPModel` from HuggingFace. CLIP backbone is frozen by default; only the CLIP projection layers (`visual_projection`, `text_projection`) and optionally the last N vision transformer blocks are trainable. Set `model.embed_dim: null` to use CLIP's native projection dimension (no extra head). Projection heads are only created when `embed_dim` differs from CLIP's native dim.

**False Negative Elimination (FNE):** During training, the dataset samples hard negatives from pre-mined caption neighbors. Neighbors with similarity > `mining.fne_threshold` are treated as false negatives and excluded. Requires mining to be run first and paths set in `mining.indices_path` / `mining.values_path`. Hard negatives are only used when `loss.use_clip_loss: false`.

**Checkpointing:** Single file `best_model.pth`, written every `save_freq` epochs (and at the final epoch). After training, this file is loaded and evaluated on the test split.

**Mixed Precision:** AMP is automatically enabled when CUDA is available.

**Optimizer:** Three separate parameter groups with different LRs: `backbone` (frozen CLIP layers if unfrozen) → `clip_proj` → `head` (custom projection). Uses AdamW with cosine LR schedule + warmup.

## Dataset Structure

```
datasets/
  flickr30k/
    flickr30k_images/        # raw images
    caption_datasets/dataset_flickr30k.json   # Karpathy JSON
    pairwise_similarities/   # mined .pt files (generated by mine_pairwise_sim.py)
  coco/
    train2014/  val2014/
    caption_datasets/dataset_coco.json
    pairwise_similarities/
```

Karpathy JSON format: `{"images": [{"split": "train|val|test|restval", "filename": "...", "sentences": [{"raw": "..."}], ...}]}`. `restval` is treated as train for Flickr30k.

## WandB Setup

Requires `~/experiments/env/wandb.env` with `WANDB_API_KEY=...` on HPC. SLURM scripts source this file. `logging.wandb_project` must be set in config. Run name is auto-generated as `{SLURM_JOB_ID}_{dataset}_{exp_name}`.
