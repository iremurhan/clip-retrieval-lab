"""
run.py
------
Main entry point for the Cross-Modal Retrieval project.
Handles configuration setup, tracker initialization, and triggers the training engine.
"""

import argparse
import torch
import logging
import wandb
import sys
import os
from transformers import CLIPTokenizer, get_cosine_schedule_with_warmup

from src.data import create_image_text_dataloader
from src.model import DualEncoder
from src.loss import SymmetricInfoNCELoss
from src.train import Trainer
from src.setup import setup_config, setup_seed, setup_tracker


def setup_logging(save_dir):
    """Configure logging to split standard info and errors."""
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    err_handler = logging.FileHandler(os.path.join(save_dir, "errors.log"), delay=True)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)
    
    return logger


def create_clip_optimizer(model, config):
    """
    Create optimizer with parameter groups for CLIP fine-tuning.
    
    Organizes parameters into three groups with different learning rates:
    1. CLIP Projections (visual_projection, text_projection) - Low LR
    2. Custom Heads (image_proj, text_proj) - Higher LR  
    3. Backbone (if unfrozen) - Very low LR
    """
    opt_name = config['training']['optimizer'].lower()
    wd = float(config['training']['weight_decay'])
    
    lr_clip_proj = float(config['training'].get('clip_projection_lr', 1e-5))
    lr_head = float(config['training'].get('head_lr', 1e-3))
    lr_backbone = float(config['training'].get('backbone_lr', 1e-6))
    
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    
    clip_proj_params = []
    custom_head_params = []
    backbone_params = []
    
    for n, p in trainable_params:
        if 'visual_projection' in n or 'text_projection' in n:
            clip_proj_params.append(p)
        elif 'image_proj' in n or 'text_proj' in n:
            custom_head_params.append(p)
        else:
            backbone_params.append(p)
            
    optimizer_grouped_parameters = [
        {'params': clip_proj_params, 'lr': lr_clip_proj},
        {'params': custom_head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': lr_backbone}
    ]
    
    print(f"Optimizer Groups Created:")
    print(f"  - CLIP Projections: {len(clip_proj_params)} tensors (LR: {lr_clip_proj})")
    print(f"  - Custom Heads:     {len(custom_head_params)} tensors (LR: {lr_head})")
    print(f"  - Backbone/Other:   {len(backbone_params)} tensors (LR: {lr_backbone})")

    if opt_name == 'adamw':
        return torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=wd)
    elif opt_name == 'adam':
        return torch.optim.Adam(optimizer_grouped_parameters, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def create_lr_scheduler(optimizer, config, num_training_steps):
    """
    Create learning rate scheduler for training.
    
    Supports cosine annealing with warmup (default) or StepLR fallback.
    """
    sched_name = config['training']['scheduler'].lower()
    epochs = config['training']['epochs']
    
    if sched_name == 'cosine':
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
    else:
        logging.warning(f"Scheduler {sched_name} not explicitly handled, defaulting to StepLR")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1) 
        
    return scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_coco.yaml', help='Path to config file')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--override', nargs='+', help='Override config params (key=value)', default=[])
    
    args = parser.parse_args()

    # 1. Load config (base + dataset-specific + CLI overrides)
    config = setup_config(args.config, args.override)
    
    # 2. Setup logging
    log_dir = config['logging']['checkpoint_dir']
    logger = setup_logging(log_dir)
    
    # 3. Debug mode check
    debug_mode = config['debug']['debug_mode']
    if debug_mode:
        logger.info("DEBUG MODE ENABLED: Disabling W&B sync, using subset.")
    
    # 4. Setup seed & device
    setup_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 5. Initialize WandB tracker
    use_wandb = setup_tracker(config, args.config, debug_mode)

    # 6. Data loaders
    logger.info("Initializing Data Loaders...")
    
    clip_model_name = config['model']['image_model_name']
    if not clip_model_name:
        raise ValueError("config['model']['image_model_name'] must be specified in config file.")
    logger.info(f"Loading CLIP Tokenizer from: {clip_model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    train_loader = create_image_text_dataloader(config, tokenizer, split='train')
    
    if debug_mode:
        logger.warning("Debug mode: Using train set as validation set.")
        val_loader = train_loader
    else:
        logger.info("Loading Validation Set...")
        val_loader = create_image_text_dataloader(config, tokenizer, split='val')

    # 7. Model & loss
    logger.info("Building Model...")
    model = DualEncoder(config).to(device)
    criterion = SymmetricInfoNCELoss(config).to(device)

    # 8. Optimizer & scheduler
    optimizer = create_clip_optimizer(model, config)
    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = create_lr_scheduler(optimizer, config, num_training_steps)

    # 9. Trainer
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, use_wandb=use_wandb
    )
    
    # 10. Resume logic
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume)
        else:
            logger.error(f"Checkpoint not found at {args.resume}")
            raise FileNotFoundError(f"Checkpoint {args.resume} does not exist")
    
    # 11. Start training
    logger.info(f"Starting training from epoch {start_epoch}...")
    try:
        trainer.fit(start_epoch=start_epoch)
        
        # 12. Final test evaluation
        logger.info("Training finished. Loading best model for Test evaluation...")
        
        best_model_path = os.path.join(config['logging']['checkpoint_dir'], "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Best model loaded successfully.")
            
            try:
                test_loader = create_image_text_dataloader(config, tokenizer, split='test')
                trainer.val_loader = test_loader
                logger.info("Running evaluation on TEST set...")
                trainer.evaluate(epoch="TEST_FINAL")
            except Exception as e:
                logger.warning(f"Could not run test evaluation (Test data missing?): {e}")
        else:
            logger.warning("Best model checkpoint not found, skipping test evaluation.")

    except KeyboardInterrupt:
        logger.info("Training interrupted manually.")
        if use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B: {e}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise e
    finally:
        if use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B: {e}")


if __name__ == "__main__":
    main()
