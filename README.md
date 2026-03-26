# CLIP-based Cross-Modal Retrieval Enhancement

CLIP-based image–text retrieval. Supports **Flickr30k** and **MS-COCO**. Optional pairwise similarity mining (caption, visual, consensus).

---

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
./scripts/start_training.sh <run_name> [config_path]
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
tez_v2_clean/
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
