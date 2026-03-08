"""
tools/eval.py
-------------
Unified evaluation script for DualEncoder models.

Two modes:
  1. Standard (default): R@K + MAP@K via src.metrics on COCO/Flickr30k.
  2. ECCV (--eccv flag): ECCV Caption protocol via eccv_caption package
     on COCO Karpathy test split (5000 images, 25000 captions).

Checkpoint handling:
    .pth file   -> load_state_dict(strict=False) onto DualEncoder
    HF model ID -> zero-shot, no weight loading (DualEncoder uses pretrained)

Usage:
    # Standard evaluation
    python tools/eval.py --config configs/config_coco.yaml \
        --checkpoint /path/to/best_model.pth --split test

    # ECCV evaluation (COCO only)
    python tools/eval.py --config configs/config_coco.yaml \
        --checkpoint /path/to/best_model.pth --eccv

    # Zero-shot
    python tools/eval.py --config configs/config_coco.yaml \
        --checkpoint openai/clip-vit-large-patch14-336 --split test
"""

import argparse
import os
import torch
import json
import logging
import numpy as np
import wandb
from tqdm import tqdm
from transformers import CLIPTokenizer

from src.setup import setup_config
from src.data import create_image_text_dataloader
from src.model import DualEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Modal Retrieval")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth or HF model name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--eccv", action="store_true", help="Run ECCV evaluation protocol (COCO only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = setup_config(config_path=args.config)

    logger.info("Loading model and tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['image_model_name'])
    model = DualEncoder(config).to(device)

    if args.checkpoint.endswith('.pth'):
        logger.info(f"Loading weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        logger.info(f"Zero-shot mode (base weights): {args.checkpoint}")

    model.eval()

    logger.info("Creating DataLoader...")
    # =========================================================
    # OOM PREVENTION SHIELD: Prevent CPU RAM & VRAM Overflow
    config['training']['batch_size'] = 32
    config['data']['num_workers'] = 2
    # =========================================================
    loader = create_image_text_dataloader(config, tokenizer, split=args.split)

    logger.info("Extracting embeddings. Please wait...")
    img_embeds_list, txt_embeds_list, image_ids_list = [], [], []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for batch in tqdm(loader, desc="Forward Pass"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)

            img_emb, txt_emb = model(images, input_ids, attn_mask)

            img_embeds_list.append(img_emb.cpu())
            txt_embeds_list.append(txt_emb.cpu())
            image_ids_list.append(batch['image_id'].cpu())

    img_embeds = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    image_ids = torch.cat(image_ids_list, dim=0)

    # Deduplicate images (keep first occurrence per image_id)
    seen_image_ids = set()
    unique_img_indices = []
    unique_image_ids_list = []

    for idx, img_id_tensor in enumerate(image_ids):
        img_id = img_id_tensor.item()
        if img_id not in seen_image_ids:
            seen_image_ids.add(img_id)
            unique_img_indices.append(idx)
            unique_image_ids_list.append(img_id)

    img_embeds_unique = img_embeds[unique_img_indices]

    # WANDB INITIALIZATION
    job_id = os.environ.get("SLURM_JOB_ID", "local_run")
    model_name = os.path.basename(args.checkpoint).replace(".pth", "").replace("/", "_")

    wandb.init(
        project="retrieval-benchmark",
        id=f"eval_{job_id}",          # Unique Run ID (Slurm Job ID)
        name=model_name,              # Readable Display Name in UI
        config=vars(args),
        resume="allow"
    )

    # ── Standard evaluation mode (R@K, MAP@K) ──
    if not args.eccv:
        logger.info("Running standard evaluation (R@K, MAP)...")
        from src.metrics import compute_recall_at_k, compute_map_at_k
        unique_image_ids = torch.tensor(unique_image_ids_list, dtype=image_ids.dtype)
        r_t2i, r_i2t = compute_recall_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids)
        map_t2i, map_i2t = compute_map_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, k_values=[5, 10])

        logger.info(
            f"\nResults:\n"
            f"  T2I: R@1: {r_t2i[1]:.2f} | R@5: {r_t2i[5]:.2f} | R@10: {r_t2i[10]:.2f}\n"
            f"  I2T: R@1: {r_i2t[1]:.2f} | R@5: {r_i2t[5]:.2f} | R@10: {r_i2t[10]:.2f}"
        )

        # W&B logging (standard mode)
        wandb.log({
            "val/t2i_r1": r_t2i[1], "val/t2i_r5": r_t2i[5], "val/t2i_r10": r_t2i[10],
            "val/i2t_r1": r_i2t[1], "val/i2t_r5": r_i2t[5], "val/i2t_r10": r_i2t[10],
            "val/t2i_map5": map_t2i[5], "val/t2i_map10": map_t2i[10],
            "val/i2t_map5": map_i2t[5], "val/i2t_map10": map_i2t[10],
        })
        wandb.finish()
        return

    # ── ECCV evaluation mode (COCO only) ──
    logger.info("Starting ECCV evaluation...")

    # Build caption ID list from Karpathy JSON to map embeddings -> COCO sentids
    captions_path = config['data']['captions_path']
    with open(captions_path, 'r') as f:
        data = json.load(f)

    caption_ids = []
    for img in data['images']:
        current_split = img['split']
        if current_split == 'restval' and args.split == 'train':
            current_split = 'train'

        if current_split == args.split:
            for sent in img['sentences'][:5]:
                caption_ids.append(sent['sentid'])

    if len(caption_ids) != len(txt_embeds):
        logger.error(f"Mismatch: Found {len(caption_ids)} captions in JSON but {len(txt_embeds)} embeddings.")
        wandb.finish()
        return

    # Compute cosine similarity matrix
    logger.info("Computing similarity matrix...")
    img_embeds_unique = torch.nn.functional.normalize(img_embeds_unique, p=2, dim=-1)
    txt_embeds = torch.nn.functional.normalize(txt_embeds, p=2, dim=-1)
    sim_matrix = torch.matmul(img_embeds_unique, txt_embeds.T).numpy()

    # Sort by descending similarity and build ECCV-format retrieval dicts
    logger.info("Sorting matrix and building ECCV retrieval dicts...")
    i2t = {}
    t2i = {}

    for i, img_id in enumerate(tqdm(unique_image_ids_list, desc="I2T Sorting")):
        sorted_caption_indices = np.argsort(-sim_matrix[i, :])
        i2t[img_id] = [caption_ids[idx] for idx in sorted_caption_indices]

    for j, cap_id in enumerate(tqdm(caption_ids, desc="T2I Sorting")):
        sorted_img_indices = np.argsort(-sim_matrix[:, j])
        t2i[cap_id] = [unique_image_ids_list[idx] for idx in sorted_img_indices]

    # Delegate to eccv_caption package
    logger.info("Calling eccv_caption library...")
    from eccv_caption import Metrics
    metric = Metrics()
    scores = metric.compute_all_metrics(
        i2t, t2i,
        target_metrics=('eccv_r1', 'eccv_map_at_r', 'eccv_rprecision', 'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls'),
        Ks=(1, 5, 10)
    )

    logger.info("====================================")
    logger.info(f"ECCV RESULTS:\n{json.dumps(scores, indent=4)}")
    logger.info("====================================")

    # W&B logging (ECCV mode)
    flat_scores = {}
    for m, values in scores.items():
        if isinstance(values, dict):
            flat_scores[f"eccv/{m}_i2t"] = values.get('i2t', 0.0)
            flat_scores[f"eccv/{m}_t2i"] = values.get('t2i', 0.0)
        else:
            flat_scores[f"eccv/{m}"] = values

    wandb.log(flat_scores)
    wandb.finish()


if __name__ == "__main__":
    main()
