"""
run.py
------
Main entry point for the Cross-Modal Retrieval project.
Handles configuration setup, tracker initialization, and triggers the training engine.
"""

import argparse
import os
import sys
import logging
import torch
import wandb
from transformers import CLIPTokenizer, get_cosine_schedule_with_warmup

from src.setup import setup_config, setup_seed, setup_tracker
from src.data import create_image_text_dataloader
from src.model import DualEncoder
from src.loss import SymmetricInfoNCELoss
from src.train import Trainer


def setup_logging():
    """Configure logging to stdout only; SLURM captures to output.log."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def create_clip_optimizer(model, config):
    """Create optimizer with parameter groups for CLIP fine-tuning."""
    opt_name = config["training"]["optimizer"].lower()
    wd = float(config["training"]["weight_decay"])
    lr_clip_proj = float(config["training"].get("clip_projection_lr", 1e-5))
    lr_head = float(config["training"].get("head_lr", 1e-3))
    lr_backbone = float(config["training"].get("backbone_lr", 1e-6))

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    clip_proj_params = []
    custom_head_params = []
    backbone_params = []

    for n, p in trainable:
        if "visual_projection" in n or "text_projection" in n:
            clip_proj_params.append(p)
        elif "image_proj" in n or "text_proj" in n:
            custom_head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if clip_proj_params:
        param_groups.append({"name": "clip_proj", "params": clip_proj_params, "lr": lr_clip_proj})
    if custom_head_params:
        param_groups.append({"name": "head", "params": custom_head_params, "lr": lr_head})
    if backbone_params:
        param_groups.append({"name": "backbone", "params": backbone_params, "lr": lr_backbone})

    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def create_lr_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler (cosine with warmup)."""
    warmup = int(config["training"].get("warmup_epochs", 2))
    warmup_steps = min(warmup * (num_training_steps // config["training"]["epochs"]), num_training_steps)
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


def main():
    parser = argparse.ArgumentParser(description="Train Cross-Modal Retrieval model")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config (e.g. configs/config_coco.yaml)")
    parser.add_argument("--override", nargs="*", default=[], help="Overrides as key=value (e.g. logging.checkpoint_dir=/path)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 1. Load config (base + dataset + overrides)
    config = setup_config(config_path=args.config, overrides=args.override)
    # Ensure data has required keys; use literal defaults only (no config-as-fallback)
    config.setdefault("data", {})
    config["data"].setdefault("max_length", 77)
    config["data"].setdefault("num_workers", 8)
    config["data"].setdefault("batch_size", 256)
    config["data"].setdefault("image_size", 336)

    # 2. Logging and seed
    setup_logging()
    log = logging.getLogger(__name__)
    setup_seed(config["training"]["seed"])

    # 3. WandB
    debug_mode = config["debug"]["debug_mode"]
    if debug_mode:
        log.info("Debug mode: using train set as validation set.")
    setup_tracker(config, debug_mode=debug_mode)

    # 4. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 5. Tokenizer and dataloaders
    model_name = config["model"]["image_model_name"]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    train_loader = create_image_text_dataloader(config, tokenizer, split="train")
    if debug_mode:
        val_loader = train_loader
    else:
        val_loader = create_image_text_dataloader(config, tokenizer, split="val")

    # 6. Model, loss, optimizer, scheduler
    model = DualEncoder(config).to(device)
    criterion = SymmetricInfoNCELoss(config)
    optimizer = create_clip_optimizer(model, config)
    num_batches = len(train_loader) * config["training"]["epochs"]
    scheduler = create_lr_scheduler(optimizer, config, num_batches)

    # 7. Trainer (use_wandb derived from config in Trainer)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # 8. Resume
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume)
            log.info(f"Resuming from epoch {start_epoch}")
        else:
            log.error(f"Checkpoint not found: {args.resume}")
            raise FileNotFoundError(args.resume)

    try:
        log.info(f"Starting training from epoch {start_epoch}...")
        trainer.fit(start_epoch=start_epoch)

        # 9. Final test evaluation (load best model, run on test set)
        log.info("Training finished. Loading best model for test evaluation...")
        best_path = os.path.join(config["logging"]["checkpoint_dir"], "best_model.pth")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            log.info("Best model loaded.")
            try:
                test_loader = create_image_text_dataloader(config, tokenizer, split="test")
                trainer.val_loader = test_loader
                log.info("Running evaluation on TEST set...")
                trainer.evaluate(epoch="TEST_FINAL")
            except Exception as e:
                log.warning(f"Test evaluation skipped: {e}")
        else:
            log.warning("Best model checkpoint not found; skipping test evaluation.")
    except KeyboardInterrupt:
        log.info("Training interrupted manually.")
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                log.warning(f"W&B finish failed: {e}")


if __name__ == "__main__":
    main()
