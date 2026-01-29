# tez_v2_clean

CLIP-based image–text retrieval (Flickr30k / COCO). Symmetric InfoNCE; optional pairwise mining.

**Setup:** `pip install -r requirements.txt`. Data under `datasets/{coco,flickr30k}/` (images + `caption_datasets/dataset_*.json`). Config: `configs/config_base.yaml` + `config_coco.yaml` / `config_flickr.yaml`.

**Training (local):** `python run.py --config configs/config_coco.yaml`  
**Training (HPC):** `./scripts/start_training.sh <run_name> [config_path]`  
Logs & checkpoints: `~/experiments/results/{coco|flickr30k}/{job_id}/`. Resume: `--resume /path/to/checkpoint.pth`.

**Mining (Flickr):** `sbatch --export=MINING_MODALITY=caption,MINING_CONFIG=configs/config_flickr.yaml scripts/mining.slurm`  
**Mining (COCO):** same with `configs/config_coco.yaml`. Modalities: `caption`, `visual`, `consensus`. Outputs: `datasets/{coco|flickr30k}/pairwise_similarities/`.

**HPC:** `mkdir -p ~/experiments/results` once. WandB: training → `retrieval-{coco|flickr30k}`; mining → `mining-{coco|flickr30k}`.

**Tools:** `diagnose_model.py` (failure report), `mine_pairwise_sim.py` (mining), `plot_failures.py`, `visualize_mining.py`.
