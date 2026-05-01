"""
scripts/eval/eval_sugarcrepe.py
-------------------------------
Standalone CLI for SugarCrepe (NeurIPS'23) compositional understanding evaluation.

Loads a saved DualEncoder checkpoint, calls the shared core in
src/eval/sugarcrepe.py, and optionally logs results to an existing WandB run
via wandb.run.summary.

For in-pipeline (no checkpoint reload) usage during training, import
`evaluate_sugarcrepe` from src.eval.sugarcrepe directly.

Usage (HPC):
    srun python scripts/eval/eval_sugarcrepe.py \
        --checkpoint checkpoints/12345_B0_coco_s42/best_model.pth \
        --wandb_run_id abc123 \
        --data_dir datasets/sugarcrepe \
        --images_dir datasets/coco/val2017

Usage (local, no WandB):
    python scripts/eval/eval_sugarcrepe.py \
        --checkpoint checkpoints/best_model.pth \
        --data_dir datasets/sugarcrepe \
        --images_dir /users/beyza.urhan/experiments/datasets/coco/val2017
"""

import argparse
import logging
import os
import sys

import torch
from transformers import CLIPTokenizer

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data import build_eval_transform
from src.eval.sugarcrepe import evaluate_sugarcrepe
from src.model import DualEncoder
from src.model_blip import DualEncoderBLIPText

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load DualEncoder from a training checkpoint (contains config + model_state_dict).

    Returns the model, config, and training epoch.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'config'. "
            "Cannot reconstruct the model."
        )

    config = checkpoint["config"]
    text_encoder_type = config.get("model", {}).get("text_encoder")
    if text_encoder_type == "blip":
        model = DualEncoderBLIPText(config).to(device)
    else:
        model = DualEncoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="SugarCrepe compositional evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume and log results")
    parser.add_argument("--data_dir", type=str, default="datasets/sugarcrepe", help="Path to SugarCrepe JSON files")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Path to COCO val2017 images (default: auto-detected)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (default: from checkpoint config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load checkpoint once and reuse config for all downstream needs.
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    image_size = config["data"]["image_size"]
    max_length = config["data"]["max_length"]

    text_encoder_type = config.get("model", {}).get("text_encoder")
    if text_encoder_type == "blip":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(config["model"]["text_model_name"])
    else:
        tokenizer = CLIPTokenizer.from_pretrained(config["model"]["image_model_name"])
    transform = build_eval_transform(image_size)

    # Resolve images directory
    if args.images_dir:
        images_dir = args.images_dir
    else:
        candidates = [
            config.get("eval", {}).get("sugarcrepe_images_dir"),
            "datasets/coco/val2017",
            "/users/beyza.urhan/experiments/datasets/coco/val2017",
        ]
        images_dir = next(
            (
                p for p in candidates
                if p and os.path.isfile(os.path.join(p, "000000000139.jpg"))
            ),
            None,
        )
        if images_dir is None:
            raise FileNotFoundError(
                "Could not find SugarCrepe COCO val2017 images. "
                f"Checked: {[p for p in candidates if p]}"
            )
        logger.info(f"Auto-detected images_dir: {images_dir}")

    # WandB resume
    if args.wandb_run_id:
        import wandb
        project = args.wandb_project
        if project is None:
            project = config.get("logging", {}).get("wandb_project", "clip-retrieval")
        wandb.init(id=args.wandb_run_id, project=project, resume="must")
        logger.info(f"Resumed WandB run: {args.wandb_run_id}")

    # Call the shared core
    results = evaluate_sugarcrepe(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device=device,
        data_dir=args.data_dir,
        images_dir=images_dir,
        max_length=max_length,
        splits=("replace", "swap", "add"),
    )

    # Log to WandB summary
    if args.wandb_run_id:
        import wandb
        for key, val in results.items():
            wandb.run.summary[f"sugarcrepe/{key}"] = val
        wandb.finish()
        logger.info("Results logged to WandB summary.")


if __name__ == "__main__":
    main()
