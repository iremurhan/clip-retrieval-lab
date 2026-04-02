"""
scripts/wise_ft.py
------------------
WiSE-FT (Weight-Space Ensembling for Fine-Tuning) post-hoc evaluation script.

Reference: Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time" (ICML 2022).

Usage (from /workspace):
    python scripts/wise_ft.py \
        --pretrained openai/clip-vit-large-patch14-336 \
        --finetuned /path/to/best_model.pth \
        --config configs/config_coco.yaml \
        --alphas 0.3 0.5 0.7 \
        --output_dir /path/to/wise_output
"""

import argparse
import logging
import os
import sys
import copy

import torch
import wandb
from transformers import CLIPModel, CLIPTokenizer

# Ensure /workspace is on the path when run inside the container
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.setup import setup_config, setup_seed, setup_tracker, _infer_dataset_name
from src.data import create_image_text_dataloader
from src.model import DualEncoder
from src.loss import build_loss
from src.train import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="WiSE-FT interpolation and evaluation")
    parser.add_argument(
        "--pretrained", type=str, required=True,
        help="HuggingFace model name, e.g. openai/clip-vit-large-patch14-336",
    )
    parser.add_argument(
        "--finetuned", type=str, required=True,
        help="Path to fine-tuned checkpoint .pth (save_checkpoint() format)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML, e.g. configs/config_coco.yaml",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
        help="Interpolation weights for fine-tuned model (0=pretrained, 1=finetuned)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save interpolated checkpoints and results",
    )
    return parser.parse_args()


def _load_pretrained_state_dict(model_name: str) -> dict:
    """
    Load the pretrained CLIP weights into a DualEncoder-compatible state dict.
    Returns only the keys under self.clip (prefixed with 'clip.').
    """
    logger.info(f"Loading pretrained CLIP weights from {model_name}...")
    clip = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    # Prefix every key with 'clip.' to match DualEncoder.state_dict() naming
    return {f"clip.{k}": v for k, v in clip.state_dict().items()}


def _interpolate(
    pretrained_sd: dict,
    finetuned_sd: dict,
    alpha: float,
) -> dict:
    """
    Compute θ_wise = (1 - α) * θ_pretrained + α * θ_finetuned for every parameter.

    - Parameters present in both: interpolated.
    - Parameters only in finetuned (e.g. custom projection heads): use finetuned directly.
    - Parameters only in pretrained: use pretrained directly.

    Args:
        pretrained_sd: State dict of pretrained model (DualEncoder key space).
        finetuned_sd:  State dict of fine-tuned model (DualEncoder key space).
        alpha:         Weight for fine-tuned model [0, 1].

    Returns:
        Interpolated state dict.
    """
    wise_sd = {}
    all_keys = set(pretrained_sd.keys()) | set(finetuned_sd.keys())

    for key in all_keys:
        if key in pretrained_sd and key in finetuned_sd:
            p = pretrained_sd[key].float()
            f = finetuned_sd[key].float()
            wise_sd[key] = (1.0 - alpha) * p + alpha * f
        elif key in finetuned_sd:
            # Fine-tuned only (e.g. custom projection heads added on top of CLIP)
            wise_sd[key] = finetuned_sd[key]
        else:
            # Pretrained only
            wise_sd[key] = pretrained_sd[key]

    return wise_sd


def _build_eval_trainer(config: dict, device: torch.device, use_wandb: bool) -> Trainer:
    """Build a Trainer with only a val_loader (no train_loader needed for evaluate())."""
    model_name = config["model"]["image_model_name"]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Mirror data keys from training config (run.py does this at runtime)
    config.setdefault("data", {})
    config["data"].setdefault("max_length", config["training"].get("max_length", 77))
    config["data"].setdefault("num_workers", config["training"].get("num_workers", 8))
    config["data"].setdefault("batch_size", config["training"].get("batch_size", 256))
    config["data"].setdefault("image_size", config["training"].get("image_size", 336))

    test_loader = create_image_text_dataloader(config, tokenizer, split="test")

    model = DualEncoder(config).to(device)
    criterion = build_loss(config)

    # Optimizer and scheduler are unused during evaluate(); provide minimal stubs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    trainer = Trainer(
        model=model,
        train_loader=test_loader,   # unused; Trainer requires it for __init__
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        use_wandb=use_wandb,
    )
    return trainer


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load fine-tuned checkpoint and the config embedded in it
    # ------------------------------------------------------------------
    logger.info(f"Loading fine-tuned checkpoint: {args.finetuned}")
    ckpt = torch.load(args.finetuned, map_location="cpu")

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {args.finetuned} does not contain 'model_state_dict'. "
            "Expected save_checkpoint() format."
        )

    finetuned_sd = ckpt["model_state_dict"]

    # The checkpoint carries the config used during training.
    ckpt_config = ckpt.get("config", {})
    original_run_id = ckpt_config.get("logging", {}).get("run_id", "unnamed")

    # ------------------------------------------------------------------
    # 2. LoRA guard
    # WiSE-FT interpolation is not directly applicable to LoRA checkpoints
    # because LoRA adds adapter weight matrices (lora_A, lora_B) that do not
    # exist in the pretrained model. Interpolating them with zeroed-out entries
    # from the pretrained side would corrupt the adapter. The correct approach
    # is to first merge LoRA weights into the base weights (W += BA * scale),
    # then interpolate. That merge step is not implemented here.
    # ------------------------------------------------------------------
    lora_rank = ckpt_config.get("model", {}).get("lora_rank", 0)
    if lora_rank and lora_rank > 0:
        logger.warning(
            "Checkpoint was trained with LoRA (lora_rank=%d). "
            "WiSE-FT interpolation is not directly applicable: LoRA adds adapter "
            "weight matrices (lora_A / lora_B) that do not exist in the pretrained "
            "model. Direct interpolation would blend fine-tuned adapters with zeros, "
            "producing incorrect weights. Merge LoRA into the base weights first, "
            "then re-run this script. Skipping.",
            lora_rank,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # 3. Load the eval config (CLI --config overrides checkpoint config for data paths)
    # ------------------------------------------------------------------
    config = setup_config(config_path=args.config)
    setup_seed(config["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 4. Build the pretrained state dict in DualEncoder key space
    # ------------------------------------------------------------------
    pretrained_sd = _load_pretrained_state_dict(args.pretrained)

    # ------------------------------------------------------------------
    # 5. Iterate over alphas
    # ------------------------------------------------------------------
    results = []   # list of (alpha, primary_score, score_dict)

    for alpha in args.alphas:
        logger.info(f"\n{'='*60}")
        logger.info(f"WiSE-FT  alpha={alpha:.2f}  ({alpha*100:.0f}% fine-tuned)")
        logger.info(f"{'='*60}")

        # 5a. Interpolate weights
        wise_sd = _interpolate(pretrained_sd, finetuned_sd, alpha)

        # 5b. Save interpolated checkpoint
        wise_ckpt_path = os.path.join(args.output_dir, f"wise_alpha{alpha:.1f}.pth")
        torch.save({"model_state_dict": wise_sd, "alpha": alpha, "config": config}, wise_ckpt_path)
        logger.info(f"Saved interpolated checkpoint: {wise_ckpt_path}")

        # 5c. Build a fresh config for this alpha.
        # - run_id stays as the original (e.g. "B0") so W&B groups all WiSE alphas together.
        # - wise_alpha is set so setup_tracker() generates the correct run name and tags.
        eval_config = copy.deepcopy(config)
        eval_config.setdefault("logging", {})["run_id"] = original_run_id
        eval_config["logging"]["wise_alpha"] = alpha
        eval_config["logging"]["checkpoint_dir"] = args.output_dir

        # 5d. Init W&B via setup_tracker (handles naming, tags, project, seed validation)
        use_wandb = eval_config["logging"].get("use_wandb", True)
        setup_tracker(eval_config)

        # 5e. Build trainer, load interpolated weights, evaluate
        trainer = _build_eval_trainer(eval_config, device, use_wandb)
        trainer.model.load_state_dict(wise_sd, strict=True)

        primary_score = trainer.evaluate(epoch="WISE_TEST")

        if use_wandb and wandb.run is not None:
            wandb.finish()

        results.append((alpha, primary_score))
        logger.info(f"alpha={alpha:.2f}  primary_score={primary_score:.2f}")

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"{'WiSE-FT Summary':^50}")
    print(f"  Checkpoint : {args.finetuned}")
    print(f"  Pretrained : {args.pretrained}")
    print("="*50)
    print(f"  {'alpha':>6}  {'primary_score':>14}")
    print(f"  {'-'*6}  {'-'*14}")
    for alpha, score in results:
        print(f"  {alpha:>6.2f}  {score:>14.2f}")
    print("="*50)


if __name__ == "__main__":
    main()
