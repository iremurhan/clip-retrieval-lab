# CLIP-based Cross-Modal Retrieval

> **Archived (November 2026).** This repository holds the code for my MSc thesis
> *Cross-Modal Retrieval: Interventions, Diagnostics, and Representational Limits* (defended June 2026).
> It is no longer maintained. `main` is the final state used for the thesis.

Image–text retrieval with CLIP (ViT-B/16 and ViT-L/14-336) on **MS-COCO** and **Flickr30k**: fine-tuning
interventions (augmentation, LLM caption rewrites, hard negatives, multi-label auxiliary loss, partial unfreezing,
SAM-based segment features), plus diagnostics (SugarCrepe, MMVP-VLM, missing positives, alignment/uniformity,
cross-dataset transfer).

## Where things are

| Path | Contents |
|---|---|
| `src/`, `run.py` | Model, data, losses, training and evaluation |
| `configs/` | Base and dataset configs; `registry*.yaml` list the thesis experiment variants |
| `scripts/` | SLURM and shell scripts (`train/`, `eval/`, `preprocess/`, `paper/` for the CLIP-B runs, `setup/` for Docker) |
| `tools/` | Mining, zero-shot evaluation, caption-rewrite utilities |
| `figs/`, `results/` | Thesis figure scripts and the generated figures, tables and data |
| `docs/methodology_reproducibility_addendum.md` | Reproducibility notes |
| `archive/`, `legacy/` | Superseded code kept for reference |

Not included: datasets, checkpoints, and the LLM caption rewrites (`caption_rewrites_*`).

## Tags

| Tag | What it marks |
|---|---|
| `thesis-final` | Final thesis code (`main` differs only in documentation) |
| `archive/consolidate-v1` | Last thesis table and qualitative-example scripts |
| `archive/b2-hard-negatives` | B2: hard-negative mining |
| `archive/b4-multilabel` | B4: multi-label classification auxiliary loss |
| `archive/b5-segment-aware`, `archive/b5-segment-only` | B5: SAM segment-map variants |
| `v1.0-baseline`, `finalized/*`, `archived/*`, `archive/i*` | Earlier experiment lines |

---

The original usage notes follow.


## Requirements

- **Python:** 3.9+ (3.10+ recommended for WandB).
- **Local run:** Install PyTorch for your CUDA version, then:
  ```bash
  pip install -r requirements.txt
  ```
  `requirements.txt` lists: `numpy`, `pandas`, `PyYAML`, `tqdm`, `transformers`, `wandb`. **Torch is not in requirements** — it is provided by the Docker image.
- **Docker (HPC / recommended):** Image `biremurhan/image-text-contrast:v0.4`. Built from `scripts/setup/Dockerfile`. Includes PyTorch; runs `pip install -r requirements.txt` internally. Build: `./scripts/setup/build.sh`.

---

## Dataset structure

Data lives under `datasets/`. Config points to `images_path` and `captions_path`.

**COCO**

```
datasets/coco/
├── train2014/
├── val2014/
└── caption_datasets/
    └── dataset_coco.json
```

**Flickr30k**

```
datasets/flickr30k/
├── flickr30k_images/
└── caption_datasets/
    └── dataset_flickr30k.json
```

**JSON format (Karpathy-style):** One JSON per dataset; top-level `{"images": [...]}`. Each image: `split` (train/val/test), `cocoid` or `imgid` or `id`, `filepath` (optional), `filename`, `sentences`: `[{"raw": "caption"}, ...]` (up to 5). For Flickr30k, `restval` is treated as train.

Download helpers: `scripts/setup/download_coco.sh`, `scripts/setup/download_flickr.sh`. COCO images: [train2014](http://images.cocodataset.org/zips/train2014.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip). Karpathy caption JSON: use pre-made `dataset_coco.json` / `dataset_flickr30k.json`. Flickr30k images: [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).

---

## Config

- Base: `configs/config_base.yaml`
- Datasets: `configs/config_coco.yaml`, `configs/config_flickr30k.yaml`
- Override example: `python run.py --config configs/config_flickr30k.yaml --override training.epochs=10`

---

## Training

**Local**

```bash
python run.py --config configs/config_coco.yaml
python run.py --config configs/config_flickr30k.yaml
```

Resume: `--resume /path/to/checkpoint.pth`

**HPC (Slurm)**

```bash
./scripts/train/start_training.sh <run_name> [config_path]
```

Logs & checkpoints: `~/experiments/results/{coco|flickr30k}/{job_id}/`

---

## Mining (pairwise similarity)

Extracts top-k neighbors (caption / visual / consensus). Outputs to `datasets/{coco|flickr30k}/pairwise_similarities/`.

**Local**

```bash
python tools/mine_pairwise_sim.py --modality caption --config configs/config_flickr30k.yaml --top_k 1000
```

**HPC (wrapper — recommended)**

```bash
# Usage: ./scripts/start_mining.sh [coco | flickr30k]
./scripts/start_mining.sh coco
./scripts/start_mining.sh flickr30k
```

**HPC (manual sbatch)**

```bash
sbatch --export=ALL,TARGET_DATASET=coco scripts/mine.slurm
sbatch --export=ALL,TARGET_DATASET=flickr30k scripts/mine.slurm
```

Logs: `~/experiments/results/{coco|flickr30k}/{job_id}/mining_log.out` (moved there after the job finishes).

---

## Project layout

```
clip-retrieval-lab/
├── configs/          # YAML configs
├── src/              # Source (data, model, loss, train)
├── tools/            # Utilities (mining, analysis)
├── scripts/          # Slurm & shell (start_mining.sh, mine.slurm)
├── run.py            # Training entry point
└── requirements.txt
```

---

## HPC & WandB setup

- **Results dir:** `mkdir -p ~/experiments/results`
- **WandB:** Requires API key. Create `~/experiments/env/wandb.env` with:
  ```
  WANDB_API_KEY=your_key_here
  ```
  The Slurm scripts mount this as `/env` and source it so mining/training can log to WandB.
- **Docker image:** `biremurhan/image-text-contrast:v0.4`. Adjust paths in `train.slurm` / `mine.slurm` for your account if needed.
