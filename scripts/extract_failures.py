"""
scripts/extract_failures.py
----------------------------
Extract worst retrieval failures from COCO 5K test set for qualitative error analysis.

For each direction (I2T, T2I), finds the 100 queries with the highest ground-truth rank
(= worst failures) and produces:
  1. An HTML report with image cards, captions, and manual annotation dropdowns.
  2. A JSON sidecar with structured data for programmatic filtering.
  3. (Optional) WandB logging: failure table + summary stats.

Usage (HPC):
    srun python scripts/extract_failures.py \
        --checkpoint /output/results/coco/12345/best_model.pth \
        --config configs/config_coco.yaml \
        --output_dir /output/results/coco/12345/failure_analysis

Usage (local):
    python scripts/extract_failures.py \
        --checkpoint checkpoints/best_model.pth \
        --config configs/config_coco.yaml \
        --output_dir failure_analysis
"""

import argparse
import base64
import json
import logging
import os
import sys
from io import BytesIO
from statistics import median

import torch
from PIL import Image
from transformers import CLIPTokenizer

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import build_eval_transform, create_image_text_dataloader
from src.metrics import _build_gt_mappings
from src.model import DualEncoder
from src.setup import setup_config, setup_seed
from src.utils import chunked_matmul

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 100

CATEGORY_OPTIONS = [
    "",
    "wrong object",
    "wrong relation",
    "wrong attribute",
    "spatial confusion",
    "counting error",
    "fine-grained confusion",
    "other",
]


# ---------------------------------------------------------------------------
# Model + embeddings
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, config, device):
    """Load DualEncoder from checkpoint, using the provided config for construction."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualEncoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """
    Mirror of Trainer._extract_embeddings — returns embeddings + metadata.

    Returns:
        img_embeds_unique: [N_imgs, D]
        txt_embeds:        [N_txts, D]
        image_ids:         [N_txts]  (image id per caption)
        unique_image_ids:  [N_imgs]
        first_occurrence_indices: list[int]
    """
    use_amp = device.type == "cuda"
    img_list, txt_list, imgid_list, sentid_list = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            img_emb, txt_emb = model(images, input_ids, attention_mask)

        img_list.append(img_emb.cpu())
        txt_list.append(txt_emb.cpu())
        imgid_list.append(batch["image_id"].cpu())
        sentid_list.append(batch["sentid"].cpu())

    img_embeds = torch.cat(img_list, dim=0)
    txt_embeds = torch.cat(txt_list, dim=0)
    image_ids = torch.cat(imgid_list, dim=0)

    seen = set()
    first_occurrence_indices = []
    unique_image_ids_list = []
    for idx in range(len(image_ids)):
        iid = image_ids[idx].item()
        if iid not in seen:
            seen.add(iid)
            first_occurrence_indices.append(idx)
            unique_image_ids_list.append(iid)

    unique_image_ids = torch.tensor(unique_image_ids_list, dtype=image_ids.dtype)
    img_embeds_unique = img_embeds[first_occurrence_indices]

    return img_embeds_unique, txt_embeds, image_ids, unique_image_ids, first_occurrence_indices


# ---------------------------------------------------------------------------
# Rank computation
# ---------------------------------------------------------------------------

def compute_gt_ranks(sims, image_ids, unique_image_ids):
    """
    Compute ground-truth ranks for both I2T and T2I directions.

    Args:
        sims: [N_imgs, N_txts] similarity matrix
        image_ids: [N_txts] image id per caption
        unique_image_ids: [N_imgs]

    Returns:
        i2t_ranks: list of (img_idx, best_rank) for each unique image
        t2i_ranks: list of (caption_idx, rank) for each caption
    """
    _, caption_to_image_idx, image_to_caption_indices = _build_gt_mappings(image_ids, unique_image_ids)

    n_imgs = sims.shape[0]
    n_txts = sims.shape[1]

    # I2T: for each image, rank of the best ground-truth caption
    i2t_ranks = []
    for i in range(n_imgs):
        gt_cap_indices = image_to_caption_indices[i]
        sorted_indices = sims[i].argsort(descending=True).tolist()
        best_rank = min(sorted_indices.index(c) for c in gt_cap_indices)
        i2t_ranks.append((i, best_rank))

    # T2I: for each caption, rank of its ground-truth image
    sims_t2i = sims.t()  # [N_txts, N_imgs]
    t2i_ranks = []
    for i in range(n_txts):
        gt_img_idx = caption_to_image_idx[i].item()
        sorted_indices = sims_t2i[i].argsort(descending=True)
        rank = (sorted_indices == gt_img_idx).nonzero(as_tuple=True)[0][0].item()
        t2i_ranks.append((i, rank))

    return i2t_ranks, t2i_ranks


# ---------------------------------------------------------------------------
# Failure extraction
# ---------------------------------------------------------------------------

def extract_failures(sims, i2t_ranks, t2i_ranks, dataset_samples,
                     unique_image_ids, image_ids, first_occurrence_indices, top_k):
    """
    Extract top_k worst failures for each direction.

    Returns:
        i2t_failures: list of dicts
        t2i_failures: list of dicts
    """
    # Sort by rank descending (worst first)
    i2t_sorted = sorted(i2t_ranks, key=lambda x: x[1], reverse=True)[:top_k]
    t2i_sorted = sorted(t2i_ranks, key=lambda x: x[1], reverse=True)[:top_k]

    unique_ids_list = unique_image_ids.tolist()

    # I2T failures
    i2t_failures = []
    for img_idx, gt_rank in i2t_sorted:
        # Ground-truth caption: pick the first caption for this image
        dataset_idx = first_occurrence_indices[img_idx]
        sample = dataset_samples[dataset_idx]

        # Top-1 retrieved caption
        top1_cap_idx = sims[img_idx].argmax().item()
        top1_sample = dataset_samples[top1_cap_idx]

        i2t_failures.append({
            "direction": "i2t",
            "image_id": unique_ids_list[img_idx],
            "gt_caption": sample["caption"],
            "top1_caption": top1_sample["caption"],
            "top1_image_id": None,
            "gt_rank": gt_rank,
            "category": "",
            "filepath": sample.get("filepath", ""),
            "filename": sample["filename"],
            "top1_filepath": None,
            "top1_filename": None,
        })

    # T2I failures
    t2i_failures = []
    sims_t2i = sims.t()  # [N_txts, N_imgs]
    for cap_idx, gt_rank in t2i_sorted:
        sample = dataset_samples[cap_idx]

        # Top-1 retrieved image
        top1_img_idx = sims_t2i[cap_idx].argmax().item()
        top1_dataset_idx = first_occurrence_indices[top1_img_idx]
        top1_sample = dataset_samples[top1_dataset_idx]

        t2i_failures.append({
            "direction": "t2i",
            "image_id": sample["image_id"],
            "gt_caption": sample["caption"],
            "top1_caption": None,
            "top1_image_id": unique_ids_list[top1_img_idx],
            "gt_rank": gt_rank,
            "category": "",
            "filepath": sample.get("filepath", ""),
            "filename": sample["filename"],
            "top1_filepath": top1_sample.get("filepath", ""),
            "top1_filename": top1_sample["filename"],
        })

    return i2t_failures, t2i_failures


# ---------------------------------------------------------------------------
# Image encoding for HTML
# ---------------------------------------------------------------------------

def img_to_base64(images_root, filepath, filename, max_size=384):
    """Load image, resize for HTML embedding, return base64 data URI."""
    if filepath:
        path = os.path.join(images_root, filepath, filename)
        if not os.path.exists(path):
            path = os.path.join(images_root, filename)
    else:
        path = os.path.join(images_root, filename)

    if not os.path.exists(path):
        return ""

    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(i2t_failures, t2i_failures, images_root, top_k=100):
    """Build self-contained HTML report with embedded images."""

    category_options_html = "\n".join(
        f'<option value="{c}">{c if c else "(select category)"}</option>'
        for c in CATEGORY_OPTIONS
    )

    def rank_style(rank):
        if rank >= 1000:
            return 'style="color:#dc2626;font-weight:bold"'
        if rank >= 100:
            return 'style="color:#ea580c;font-weight:bold"'
        return ""

    def card_i2t(entry, idx):
        img_b64 = img_to_base64(images_root, entry["filepath"], entry["filename"])
        rs = rank_style(entry["gt_rank"])
        return f"""
        <div class="card">
          <div class="card-header">
            <span class="idx">#{idx+1}</span>
            <span class="direction-badge i2t">I2T</span>
            <span>Image ID: <b>{entry['image_id']}</b></span>
            <span {rs}>GT Rank: <b>{entry['gt_rank']}</b></span>
          </div>
          <div class="card-body">
            <div class="img-col">
              <img src="{img_b64}" alt="query image" />
              <div class="img-label">Query Image</div>
            </div>
            <div class="text-col">
              <div class="caption-block">
                <div class="label">Ground-Truth Caption:</div>
                <div class="caption">{entry['gt_caption']}</div>
              </div>
              <div class="caption-block">
                <div class="label">Model Top-1 Caption:</div>
                <div class="caption top1">{entry['top1_caption']}</div>
              </div>
              <div class="annotation">
                <label>Category:</label>
                <select data-idx="{idx}" data-dir="i2t">{category_options_html}</select>
              </div>
            </div>
          </div>
        </div>"""

    def card_t2i(entry, idx):
        gt_b64 = img_to_base64(images_root, entry["filepath"], entry["filename"])
        top1_b64 = img_to_base64(images_root, entry["top1_filepath"], entry["top1_filename"])
        rs = rank_style(entry["gt_rank"])
        return f"""
        <div class="card">
          <div class="card-header">
            <span class="idx">#{idx+1}</span>
            <span class="direction-badge t2i">T2I</span>
            <span>Image ID: <b>{entry['image_id']}</b></span>
            <span {rs}>GT Rank: <b>{entry['gt_rank']}</b></span>
          </div>
          <div class="card-body">
            <div class="text-col" style="flex:1 1 100%;margin-bottom:8px;">
              <div class="caption-block">
                <div class="label">Query Caption:</div>
                <div class="caption">{entry['gt_caption']}</div>
              </div>
            </div>
          </div>
          <div class="card-body">
            <div class="img-col">
              <img src="{gt_b64}" alt="ground truth image" />
              <div class="img-label">Ground-Truth Image (ID: {entry['image_id']})</div>
            </div>
            <div class="img-col">
              <img src="{top1_b64}" alt="top-1 retrieved image" />
              <div class="img-label">Model Top-1 Image (ID: {entry['top1_image_id']})</div>
            </div>
          </div>
          <div class="card-body">
            <div class="annotation">
              <label>Category:</label>
              <select data-idx="{idx}" data-dir="t2i">{category_options_html}</select>
            </div>
          </div>
        </div>"""

    i2t_cards = "\n".join(card_i2t(e, i) for i, e in enumerate(i2t_failures))
    t2i_cards = "\n".join(card_t2i(e, i) for i, e in enumerate(t2i_failures))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Retrieval Failure Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #f5f5f5; padding: 24px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 8px; }}
  .meta {{ color: #666; margin-bottom: 24px; font-size: 14px; }}
  h2 {{ margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 2px solid #e5e5e5; }}
  .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
  .card-header {{ display: flex; gap: 16px; align-items: center; padding: 10px 16px; background: #fafafa; border-bottom: 1px solid #eee; font-size: 14px; }}
  .card-body {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 16px; }}
  .idx {{ font-weight: bold; color: #888; }}
  .direction-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; color: #fff; }}
  .direction-badge.i2t {{ background: #2563eb; }}
  .direction-badge.t2i {{ background: #7c3aed; }}
  .img-col {{ text-align: center; }}
  .img-col img {{ max-width: 320px; max-height: 320px; border-radius: 4px; border: 1px solid #ddd; }}
  .img-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
  .text-col {{ flex: 1; min-width: 280px; }}
  .caption-block {{ margin-bottom: 12px; }}
  .label {{ font-size: 12px; font-weight: 600; color: #666; margin-bottom: 2px; }}
  .caption {{ font-size: 14px; line-height: 1.5; padding: 8px; background: #f9fafb; border-radius: 4px; }}
  .caption.top1 {{ background: #fef2f2; border-left: 3px solid #ef4444; }}
  .annotation {{ margin-top: 8px; font-size: 14px; }}
  .annotation select {{ padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; }}
</style>
</head>
<body>
<h1>Retrieval Failure Analysis</h1>
<p class="meta">Top {top_k} worst failures per direction, sorted by GT rank (worst first).</p>

<h2>Image-to-Text Failures (I2T)</h2>
{i2t_cards}

<h2>Text-to-Image Failures (T2I)</h2>
{t2i_cards}

</body>
</html>"""


# ---------------------------------------------------------------------------
# WandB logging
# ---------------------------------------------------------------------------

def log_to_wandb(i2t_failures, t2i_failures, wandb_run_id, wandb_project):
    """Resume WandB run and log failure table + summary stats."""
    import wandb

    wandb.init(id=wandb_run_id, project=wandb_project, resume="must")
    logger.info(f"Resumed WandB run: {wandb_run_id}")

    # Table
    columns = ["direction", "image_id", "gt_caption", "top1_caption",
               "top1_image_id", "gt_rank", "category"]
    table = wandb.Table(columns=columns)
    for entry in i2t_failures + t2i_failures:
        table.add_data(
            entry["direction"],
            entry["image_id"],
            entry["gt_caption"],
            entry.get("top1_caption", ""),
            entry.get("top1_image_id", ""),
            entry["gt_rank"],
            entry["category"],
        )
    wandb.log({"failure_analysis/failures": table}, commit=False)

    # Summary stats
    i2t_gt_ranks = [e["gt_rank"] for e in i2t_failures]
    t2i_gt_ranks = [e["gt_rank"] for e in t2i_failures]
    wandb.run.summary["failure_analysis/median_gt_rank_i2t"] = median(i2t_gt_ranks)
    wandb.run.summary["failure_analysis/median_gt_rank_t2i"] = median(t2i_gt_ranks)

    wandb.finish()
    logger.info("Failure analysis logged to WandB.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract worst retrieval failures from COCO 5K test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset config (e.g. configs/config_coco.yaml)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HTML + JSON")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of worst failures per direction")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project (default: from config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Config
    config = setup_config(config_path=args.config, overrides=[])
    setup_seed(config["training"]["seed"])

    # Model
    model = load_model(args.checkpoint, config, device)
    model_name = config["model"]["image_model_name"]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Test dataloader
    test_loader = create_image_text_dataloader(config, tokenizer, split="test")
    dataset = test_loader.dataset

    # Extract embeddings
    logger.info("Extracting embeddings on test set...")
    img_embeds_unique, txt_embeds, image_ids, unique_image_ids, first_occ = \
        extract_embeddings(model, test_loader, device)
    logger.info(f"Embeddings: {img_embeds_unique.shape[0]} images, {txt_embeds.shape[0]} captions")

    # Similarity matrix: [N_imgs, N_txts]
    sims = chunked_matmul(img_embeds_unique, txt_embeds)

    # Compute GT ranks
    logger.info("Computing ground-truth ranks...")
    i2t_ranks, t2i_ranks = compute_gt_ranks(sims, image_ids, unique_image_ids)

    # Extract failures
    i2t_failures, t2i_failures = extract_failures(
        sims, i2t_ranks, t2i_ranks, dataset.samples,
        unique_image_ids, image_ids, first_occ, args.top_k,
    )

    logger.info(f"I2T worst failure rank range: {i2t_failures[0]['gt_rank']} - {i2t_failures[-1]['gt_rank']}")
    logger.info(f"T2I worst failure rank range: {t2i_failures[0]['gt_rank']} - {t2i_failures[-1]['gt_rank']}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # JSON sidecar
    json_path = os.path.join(args.output_dir, "failures.json")
    json_data = {"i2t": i2t_failures, "t2i": t2i_failures}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"JSON saved: {json_path}")

    # HTML report
    images_root = config["data"]["images_path"]
    html_path = os.path.join(args.output_dir, "failures.html")
    html = build_html(i2t_failures, t2i_failures, images_root, top_k=args.top_k)
    with open(html_path, "w") as f:
        f.write(html)
    logger.info(f"HTML saved: {html_path}")

    # WandB
    if args.wandb_run_id:
        project = args.wandb_project or config.get("logging", {}).get("wandb_project", "clip-retrieval")
        log_to_wandb(i2t_failures, t2i_failures, args.wandb_run_id, project)


if __name__ == "__main__":
    main()
