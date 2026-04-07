"""
scripts/eval_sugarcrepe.py
--------------------------
SugarCrepe (NeurIPS'23) compositional understanding evaluation.

For each sub-category, loads (image, positive caption, negative caption) triplets,
computes cosine similarity via the fine-tuned DualEncoder, and measures accuracy
(positive caption scored higher than negative).

Results are logged to an existing WandB run via wandb.run.summary.

Usage (HPC):
    srun python scripts/eval_sugarcrepe.py \
        --checkpoint checkpoints/12345_B0_coco_s42/best_model.pth \
        --wandb_run_id abc123 \
        --data_dir datasets/sugarcrepe

Usage (local, no WandB):
    python scripts/eval_sugarcrepe.py \
        --checkpoint checkpoints/best_model.pth \
        --data_dir datasets/sugarcrepe
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import DualEncoder
from src.data import build_eval_transform

logger = logging.getLogger(__name__)

SUBCATEGORIES = [
    "add_att", "add_obj",
    "replace_att", "replace_obj", "replace_rel",
    "swap_att", "swap_obj",
]


def load_model_from_checkpoint(checkpoint_path, device):
    """Load DualEncoder from a training checkpoint (contains config + model_state_dict)."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'config'. "
            "Cannot reconstruct the model."
        )

    config = checkpoint["config"]
    model = DualEncoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_name = config["model"]["image_model_name"]
    image_size = config["data"]["image_size"]
    max_length = config["data"]["max_length"]

    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    return model, model_name, image_size, max_length


def evaluate_subcategory(model, tokenizer, transform, data, images_dir, max_length, device):
    """
    Evaluate a single SugarCrepe sub-category.

    Args:
        data: dict keyed by string indices, each with 'filename', 'caption', 'negative_caption'.

    Returns:
        accuracy (float): fraction of triplets where sim(image, pos) > sim(image, neg).
    """
    correct = 0
    total = 0

    for entry in data.values():
        filename = entry["filename"]
        pos_caption = entry["caption"]
        neg_caption = entry["negative_caption"]

        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            # SugarCrepe uses bare IDs (e.g. 000000085329.jpg) but COCO images
            # are prefixed with COCO_val2014_.  Try the prefixed name as fallback.
            prefixed = os.path.join(images_dir, f"COCO_val2014_{filename}")
            if os.path.isfile(prefixed):
                img_path = prefixed
            else:
                logger.warning(f"Image not found, skipping: {img_path}")
                continue

        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

        tok_pos = tokenizer(
            pos_caption, return_tensors="pt",
            max_length=max_length, padding="max_length", truncation=True,
        )
        tok_neg = tokenizer(
            neg_caption, return_tensors="pt",
            max_length=max_length, padding="max_length", truncation=True,
        )

        pos_ids = tok_pos["input_ids"].to(device)           # [1, L]
        pos_mask = tok_pos["attention_mask"].to(device)      # [1, L]
        neg_ids = tok_neg["input_ids"].to(device)            # [1, L]
        neg_mask = tok_neg["attention_mask"].to(device)      # [1, L]

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            img_emb = model.encode_image(img_tensor)                   # [1, D]
            pos_emb = model.encode_text(pos_ids, pos_mask)             # [1, D]
            neg_emb = model.encode_text(neg_ids, neg_mask)             # [1, D]

        sim_pos = F.cosine_similarity(img_emb, pos_emb).item()
        sim_neg = F.cosine_similarity(img_emb, neg_emb).item()

        if sim_pos > sim_neg:
            correct += 1
        total += 1

    if total == 0:
        raise RuntimeError("No valid samples found. Check images_dir and data files.")
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="SugarCrepe compositional evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume and log results")
    parser.add_argument("--data_dir", type=str, default="datasets/sugarcrepe", help="Path to SugarCrepe JSON files")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Path to COCO val2014 images (default: inferred from checkpoint config)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (default: from checkpoint config)")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model, model_name, image_size, max_length = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    transform = build_eval_transform(image_size)

    # Resolve images directory
    if args.images_dir:
        images_dir = args.images_dir
    else:
        # Checkpoint config stores data.images_path (e.g. "datasets/coco")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        base_images_path = ckpt["config"]["data"]["images_path"]
        images_dir = os.path.join(base_images_path, "val2014")
        logger.info(f"Inferred images_dir from config: {images_dir}")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # WandB resume
    if args.wandb_run_id:
        import wandb
        project = args.wandb_project
        if project is None:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            project = ckpt["config"].get("logging", {}).get("wandb_project", "clip-retrieval")
        wandb.init(id=args.wandb_run_id, project=project, resume="must")
        logger.info(f"Resumed WandB run: {args.wandb_run_id}")

    # Evaluate each sub-category
    results = {}
    for subcat in SUBCATEGORIES:
        json_path = os.path.join(args.data_dir, f"{subcat}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"SugarCrepe data file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        acc = evaluate_subcategory(model, tokenizer, transform, data, images_dir, max_length, device)
        results[subcat] = acc
        logger.info(f"  {subcat}: {acc:.4f} ({len(data)} samples)")

    # Overall accuracy
    overall = sum(results.values()) / len(results)
    results["overall"] = overall
    logger.info(f"  overall: {overall:.4f}")

    # Log to WandB summary
    if args.wandb_run_id:
        import wandb
        for key, val in results.items():
            wandb.run.summary[f"sugarcrepe/{key}"] = val
        wandb.finish()
        logger.info("Results logged to WandB summary.")


if __name__ == "__main__":
    main()
