# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIP-based cross-modal image–text retrieval. Fine-tunes `openai/clip-vit-large-patch14-336` on **Flickr30k** and **MS-COCO** through a registry of ablations:
- **B0** — InfoNCE baseline, no augmentation, no intra-modal contrast
- **B0+** — full-featured baseline: intra-modal contrast (image-image between
  two augmented views, text-text between two paraphrases), retrieval-safe
  augmentation magnitudes, paraphrase positives sampled from precomputed JSON
- **B0_uf5/6/7** — B0 with deeper vision-block unfreezing
- **B1** — SigLIP loss
- **B2** — NegCLIP-style syntactic hard negatives (spaCy POS-tag swap)
- **B4** — multi-label classification auxiliary loss (COCO categories)
- **B5a / B5b / B5c** — segment-aware patch enrichment (spatial / semantic / continuous; SAM-precomputed)

All ablations are selectable via `configs/registry.yaml` entries.

## Environment Setup

```bash
# Activate the local venv (Python 3.13)
source venv/bin/activate

# Install dependencies (torch is NOT in requirements.txt — provide it separately)
pip install -r requirements.txt
```

Docker image for HPC: `biremurhan/image-text-contrast:v0.16` (includes PyTorch, vllm, gcc).

## Common Commands

**Training (local)**
```bash
python run.py --config configs/config_flickr30k.yaml
python run.py --config configs/config_coco.yaml

# Run a registry-named ablation
python run.py --config configs/config_coco.yaml --run B5a_seg_spatial

# With CLI overrides (highest priority)
python run.py --config configs/config_flickr30k.yaml --override training.epochs=10

# Resume from checkpoint
python run.py --config configs/config_flickr30k.yaml --resume checkpoints/best_model.pth
```

**Training (HPC/SLURM)**
```bash
./scripts/train/start_training.sh <run_name> [config_path]
```

**Debug mode (fast iteration, uses train set for val)**
```bash
python run.py --config configs/config_flickr30k.yaml --override debug.debug_mode=true debug.debug_samples=100
```

## Config System

Configs are hierarchical YAML files merged at runtime:

1. `configs/config_base.yaml` — all defaults (model, loss, training, augment, paraphraser, logging)
2. `configs/config_{coco,flickr30k}.yaml` — dataset-specific paths (images, captions, paraphraser rewrites, B5 `seg_map_dir` / `sam_feature_dir`, B4 `cls_label_path`); deep-merged on top of base
3. `configs/registry.yaml` — named ablation entries. Optional `parent:` field per entry resolves recursively (depth-first, parent → child). Selected via `--run <name>`.
4. `--override key.subkey=value` — CLI overrides applied last (highest priority).

`setup_config()` and `load_registry_overrides()` in `src/setup.py` handle the merge.

WandB **requires** `logging.wandb_project` to be set in the config (or via override) — if missing, WandB is silently skipped. Optional `logging.lineage` dict is forwarded to WandB as both flat `lineage.*` keys and a nested `lineage` form for cross-run grouping.

## Architecture

```
run.py                  Training entry point: config → dataloader → DualEncoder → Trainer.fit().
                        Owns optimizer param-group routing and the per-group LR scheduler.
src/
  setup.py              Config loading (deep-merge + CLI overrides + registry parent inheritance),
                        WandB init (with logging.lineage forwarding), seed setup.
  data.py               CaptionImageDataset (Karpathy JSON), __getitem__ with
                        emit_aug_views gate (skipped when intra_img_weight=0),
                        HardNegativeGenerator (B2), SegmentFeatureLoader/SAMFeatureLoader (B5),
                        DataLoader factory.
  model.py              DualEncoder: CLIP backbone + selective freezing + B4 cls_head + B5
                        seg_embedding/seg_projection plus B5d fusion_module.
  loss.py               SymmetricInfoNCELoss + SigLIPLoss + build_loss factory; both losses
                        accept pair args (img/txt _aug_a/b) and NegCLIP-asymmetric hard-neg.
  paraphraser.py        PrecomputedLLMParaphraser: generate() (single rewrite lookup) +
                        sample_pair() (two distinct rewrites per sentid, no runtime LLM).
  train.py              Trainer: train_epoch, evaluate, checkpoint save/load, WandB logging.
  grad_cache.py         Gradient caching for large effective batch sizes (inter-modal only).
  metrics.py            compute_recall_at_k, compute_map_at_k (Recall@1/5/10, MAP@5/10).
  utils.py              compute_grad_norm and misc helpers.
  eval/
    sugarcrepe.py       evaluate_sugarcrepe() — model-callable core for SugarCrepe metrics.
tools/
  build_coco_multilabel.py  COCO multilabel target builder (B4 prep).
  eval_zero_shot.py         Zero-shot evaluation tool.
configs/
  config_base.yaml      All defaults.
  config_{coco,flickr30k}.yaml   Dataset-specific paths (images, captions, paraphraser rewrites,
                                  seg_map_dir, sam_feature_dir, cls_label_path).
  registry.yaml         Named ablation entries with parent: inheritance and research hypotheses.
scripts/
  train/                Training and experiment launching.
    train.slurm         Unified SLURM training job (merged train + ablation).
    resume.slurm        Resume from existing checkpoint + WandB run.
    launch_run.sh       Batch SLURM submission (multi-dataset, multi-seed).
    launch_ablation.sh  Single ablation job submission.
    start_training.sh   Simple HPC training wrapper.
  eval/                 Evaluation scripts.
    eval_sugarcrepe.py  SugarCrepe CLI (loads checkpoint, calls src/eval/sugarcrepe.py).
    eval_sugarcrepe.slurm  SLURM wrapper for SugarCrepe eval.
    eval_zero_shot.slurm   Zero-shot CLIP retrieval eval (rgb / mask modes).
    extract_failures.py    Worst-failure extraction + HTML report.
    extract_failures.slurm SLURM wrapper for failure analysis.
  preprocess/           Data preprocessing pipelines.
    precompute_sam.py   SAM masks/segments (B5a/b/c) and encoder features (B5d).
    precompute_sam.slurm       SAM mask/encoder-feature SLURM array job.
    precompute_sam_stage2.slurm  SAM stage 2 per-mode post-processing.
  setup/                Environment, Docker, and dataset setup.
    Dockerfile, build.sh, download_coco.sh, download_flickr.sh,
    build_multilabel.slurm
  util/                 Shared helpers and maintenance scripts.
    detect_dataset.sh   Dataset name extraction from config YAML.
    wandb_cleanup.sh    WandB sync + server-side verify + cleanup.
    wandb_sync.slurm    Manual WandB sync for crashed/offline runs.
legacy/                 Archived (paraphrase generation slurm, etc.) — not on the active training path.
```

### Key Design Decisions

**DualEncoder (`src/model.py`):** Wraps `CLIPModel` from HuggingFace. CLIP backbone is frozen by default; only the CLIP projection layers (`visual_projection`, `text_projection`) and optionally the last N vision transformer blocks are trainable. Set `model.embed_dim: null` to use CLIP's native projection dimension (no extra head).

**Intra-modal contrast (`src/data.py`, `src/loss.py`, `src/train.py`):** The inter-modal contrast uses a clean image (eval-style transform) and the original caption — these anchors are never augmented or paraphrased. The intra-modal image-image contrast is computed between two independent augmented views (`image_aug_a`, `image_aug_b`), gated on `loss.intra_img_weight > 0` (when off, `transform_aug` is not built and the keys are not emitted — feature-off / no work). The intra-modal text-text contrast uses two distinct paraphrases per sentid sampled without replacement from the precomputed JSON via `paraphraser.sample_pair()`. No runtime LLM generation. No `random.choice([orig, *rewrites])` mixing in `__getitem__` (the previous design polluted the inter-modal text anchor; removed in the consolidation).

**Augmentation magnitudes:** `aug_crop_scale_min=0.7`, `color_jitter_strength=0.2`, `use_grayscale=false`. These are the retrieval-safe magnitudes promoted into base from the historical B0v3 ablation; runs under the previous defaults (0.4 / 0.4 / true) are tagged `lineage.aug_magnitudes:v1`, runs under the current defaults are tagged `v2` via `logging.lineage.aug_magnitudes` (forwarded to WandB as both a config key and a filterable run tag).

**B2 hard negatives (`src/data.py::HardNegativeGenerator`):** spaCy POS-tag word-swap against a per-batch noun/verb/adjective pool, with a word-order shuffle fallback. Loss extends i2t similarity to N×2N (NegCLIP-asymmetric: t2i stays N×N because there are no negative images by construction).

**B5 segment-aware patches (`src/model.py`, `src/data.py::SegmentFeatureLoader`, `scripts/preprocess/precompute_sam.py`):** Three modes — `spatial` (28-bin discrete), `semantic` (81-class discrete; COCO categories), `continuous` (5-d MLP-projected). Per-patch tensors are precomputed offline (two-stage SAM pipeline) and packed COW-safe in shared memory. Forward path injects the segment embedding into `vm.embeddings(...)` output before the encoder; CLS gets a zero vector. Augmented views skip seg injection (RandomResizedCrop breaks alignment).

**B5d multistream fusion (`src/model.py`, `src/data.py::SAMFeatureLoader`, `scripts/preprocess/precompute_sam.py --encoder_features`):** `seg_mode: multistream` loads frozen SAM ViT-B image-encoder features (`[256, pool, pool]`, usually pool=8) and fuses them with CLIP's post-encoder CLS token before `visual_projection`. Fusion types are `gate`, `cross_attn`, and `concat_proj`. This path does not inject patch embeddings and does not enable CLIP gradient checkpointing.

**B4 multi-label aux loss (`src/model.py::cls_head`, `src/train.py`):** Linear head on `pooler_output`, BCE-with-logits against precomputed COCO category targets. Strict COCO-only (`data.dataset == 'coco'` enforced at Trainer init). Mutually exclusive with B5.

**Checkpointing:** Single file `best_model.pth`, written every `save_freq` epochs (and at the final epoch). After training, this file is loaded and evaluated on the test split.

**Mixed Precision:** AMP (bfloat16) is automatically enabled when CUDA is available.

**Optimizer (`run.py`):** Six named parameter groups with separate LRs:
`clip_proj` (cosine) → `cls_head` (constant, B4) → `seg_embedding` (constant, B5a/b) → `seg_projection` (constant, B5c) → `fusion_module` (constant, B5d) → `backbone` (cosine).
Constant groups (in `_CONSTANT_LR_GROUPS`) keep their base LR throughout training; cosine groups follow the standard cosine-with-warmup schedule via `LambdaLR`. Implementation lives in `create_clip_optimizer` and `create_lr_scheduler`.

## Dataset Structure

```
datasets/
  flickr30k/
    flickr30k_images/                            # raw images
    caption_datasets/dataset_flickr30k.json      # Karpathy JSON
    caption_rewrites_*.json                      # LLM paraphrases (paraphraser.paths)
    sam_segments/                                # B5 precomputed: spatial_bins.pt / semantic_ids.pt / continuous_features.pt
    sam_encoder_features/                        # B5d precomputed: sam_encoder_features_pool8.pt
  coco/
    train2014/  val2014/
    caption_datasets/dataset_coco.json
    caption_rewrites_*.json
    sam_segments/
    sam_encoder_features/
    coco_multilabel_map.pt                       # B4 precomputed (tools/build_coco_multilabel.py)
```

Karpathy JSON format: `{"images": [{"split": "train|val|test|restval", "filename": "...", "sentences": [{"raw": "..."}], ...}]}`. `restval` is treated as train for Flickr30k.

## WandB Setup

Requires `~/experiments/env/wandb.env` with `WANDB_API_KEY=...` on HPC. SLURM scripts source this file. `logging.wandb_project` must be set in config. Run name is auto-generated as `{SLURM_JOB_ID}_{dataset}_{exp_name}`.
