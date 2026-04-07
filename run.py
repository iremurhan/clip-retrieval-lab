"""
run.py
------
Main entry point for the Cross-Modal Retrieval project.
Handles configuration setup, tracker initialization, and triggers the training engine.
"""

import argparse
import os
import signal
import sys
import logging
import warnings
import torch
import wandb
from transformers import CLIPTokenizer, get_cosine_schedule_with_warmup

from src.setup import setup_config, setup_seed, setup_tracker, load_registry_overrides
from src.data import create_image_text_dataloader
from src.model import DualEncoder
from src.loss import build_loss
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
    lr_backbone = float(config["training"].get("backbone_lr", 1e-6))

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    clip_proj_params = []
    custom_head_params = []
    cls_head_params = []
    backbone_params = []

    for n, p in trainable:
        if "visual_projection" in n or "text_projection" in n:
            clip_proj_params.append(p)
        elif "image_proj" in n or "text_proj" in n:
            custom_head_params.append(p)
        elif "cls_head" in n:
            cls_head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if clip_proj_params:
        param_groups.append({"name": "clip_proj", "params": clip_proj_params, "lr": lr_clip_proj})
    if custom_head_params:
        raise RuntimeError(
            "custom_head_params is non-empty but embed_dim is expected to be null. "
            "This indicates image_proj/text_proj layers exist unexpectedly. "
            "Check config['model']['embed_dim']."
        )
    if cls_head_params:
        param_groups.append({"name": "cls_head", "params": cls_head_params, "lr": lr_clip_proj})
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
    # Suppress the spurious "scheduler.step() before optimizer.step()" warning that
    # transformers' get_cosine_schedule_with_warmup triggers internally during __init__.
    # The training loop calls optimizer.step() before scheduler.step() correctly.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
        warnings.filterwarnings("ignore", message=".*Detected call of.*lr_scheduler.step.*")
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


def main():
    parser = argparse.ArgumentParser(description="Train Cross-Modal Retrieval model")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config (e.g. configs/config_coco.yaml)")
    parser.add_argument("--override", nargs="*", default=[], help="Overrides as key=value (e.g. logging.checkpoint_dir=/path)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run", type=str, default=None, help="Run ID from configs/registry.yaml (e.g. B0, FULL)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override (sets training.seed)")
    args = parser.parse_args()

    # 1. Load config (base + dataset + registry overrides + CLI overrides)
    # Order: base < dataset config < registry overrides < --override (highest priority)
    registry_overrides = []
    if args.run:
        registry_overrides = load_registry_overrides(args.run)
    all_overrides = registry_overrides + (args.override or [])
    config = setup_config(config_path=args.config, overrides=all_overrides)

    # Set logging.run_id from --run if provided
    if args.run:
        config.setdefault("logging", {})["run_id"] = args.run

    # Apply --seed override (after registry so it takes highest priority)
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed

    # Build checkpoint dir: checkpoints/{SLURM_JOB_ID}_{run_id}_{dataset}_s{seed}/
    # Only when checkpoint_dir has not already been set via --override
    if not any("logging.checkpoint_dir" in o for o in (args.override or [])):
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_id = config.get("logging", {}).get("run_id", "unnamed")
        dataset = config["data"]["dataset"]
        seed = config["training"]["seed"]
        config["logging"]["checkpoint_dir"] = f"checkpoints/{job_id}_{run_id}_{dataset}_s{seed}"

    # 2. Logging and seed
    setup_logging()
    log = logging.getLogger(__name__)
    setup_seed(config["training"]["seed"])

    # 3. WandB
    setup_tracker(config)

    # 4. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 5. Tokenizer and dataloaders
    model_name = config["model"]["image_model_name"]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    train_loader = create_image_text_dataloader(config, tokenizer, split="train")
    val_loader = create_image_text_dataloader(config, tokenizer, split="val")

    # 6. Model, loss, optimizer, scheduler
    model = DualEncoder(config).to(device)
    criterion = build_loss(config)
    optimizer = create_clip_optimizer(model, config)
    if hasattr(criterion, 'bias') and isinstance(criterion.bias, torch.nn.Parameter):
        optimizer.add_param_group({
            'params': [criterion.bias],
            'lr': config['training']['clip_projection_lr'],
            'name': 'siglip_bias',
        })
        log.info(f"SigLIP bias added to optimizer (lr={config['training']['clip_projection_lr']})")
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
        clip_tokenizer=tokenizer,
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

    signal.signal(signal.SIGTERM, lambda s, f: setattr(trainer, '_sigterm_received', True))

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"{args.run or 'unnamed'}_{config['data']['dataset']}_s{config['training']['seed']}"

    try:
        log.info(f"Starting training from epoch {start_epoch}...")
        trainer.fit(start_epoch=start_epoch)
        if wandb.run is not None:
            wandb.alert(
                title=f"Job done: {run_name}",
                text=(
                    f"Job ID: {job_id}\n"
                    f"Run: {run_name}\n"
                    f"Checkpoint: {config['logging']['checkpoint_dir']}"
                ),
                level=wandb.AlertLevel.INFO,
            )
    except KeyboardInterrupt:
        log.info("Training interrupted manually.")
    except Exception as e:
        if wandb.run is not None:
            wandb.run.summary["status"] = "failed"
            wandb.run.summary["error_type"] = type(e).__name__
            wandb.run.summary["error_message"] = str(e)
            wandb.alert(
                title=f"Job FAILED: {run_name}",
                text=(
                    f"Job ID: {job_id}\n"
                    f"Run: {run_name}\n"
                    f"Error: {type(e).__name__}: {e}"
                ),
                level=wandb.AlertLevel.ERROR,
            )
        log.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if wandb.run is not None:
            try:
                wandb.finish(timeout=30)
            except Exception as e:
                log.warning(f"wandb.finish() failed: {e}")


if __name__ == "__main__":
    main()
