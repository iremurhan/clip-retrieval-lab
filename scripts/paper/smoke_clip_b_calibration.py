#!/usr/bin/env python3
"""
Container smoke tests for the paper CLIP-B baseline-calibration workflow.
"""

import copy
import json
import os
import tempfile
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPTokenizer

from run import create_clip_optimizer, create_lr_scheduler
from src.data import create_image_text_dataloader
from src.loss import build_loss
from src.model import DualEncoder
from src.setup import load_registry_overrides, setup_config, setup_seed
from src.train import Trainer
from tools.eval_paper_zero_shot import (
    assert_unchanged,
    encode_images,
    encode_texts,
    parameter_fingerprints,
)
from src.utils import chunked_matmul
from tools.eval_zero_shot import compute_standard_metrics


PAPER_REGISTRY = "configs/registry_paper_clip_b.yaml"
THESIS_REGISTRY = "configs/registry.yaml"
CONFIG = "configs/config_coco.yaml"


def load_config(run_id, registry=PAPER_REGISTRY, **extra):
    overrides = load_registry_overrides(run_id, registry_path=registry)
    for key, value in extra.items():
        overrides.append(f"{key}={value}")
    cfg = setup_config(config_path=CONFIG, overrides=overrides)
    cfg.setdefault("logging", {})["registry_file"] = registry
    cfg.setdefault("logging", {})["registry_id"] = run_id
    cfg.setdefault("logging", {})["run_id"] = run_id
    cfg["data"]["num_workers"] = 0
    cfg["training"]["batch_size"] = 4
    cfg["debug"]["debug_mode"] = True
    cfg["debug"]["debug_samples"] = 8
    cfg["logging"]["use_wandb"] = False
    cfg["eval"]["ood_retrieval"]["enabled"] = False
    cfg["eval"]["sugarcrepe"]["enabled"] = False
    cfg["eval"]["mmvp_vlm"]["enabled"] = False
    return cfg


def smoke_zero_shot(run_id, expected_resolution, expected_grid, expected_interp, device):
    cfg = load_config(run_id)
    setup_seed(cfg["training"]["seed"])
    model_id = cfg["model"]["image_model_name"]
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).float().to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    patch_size = int(model.config.vision_config.patch_size)
    assert patch_size == 16, patch_size
    assert expected_resolution // patch_size == expected_grid
    assert bool(cfg["model"].get("positional_interpolation", False)) is expected_interp

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    loader = create_image_text_dataloader(cfg, tokenizer, split="val")
    before = parameter_fingerprints(model)
    img_embeds, unique_image_ids, observed_shape = encode_images(model, loader, device, expected_interp)
    txt_embeds, image_ids, _ = encode_texts(model, loader, device)
    sims = chunked_matmul(img_embeds, txt_embeds)
    metrics = compute_standard_metrics(img_embeds, txt_embeds, image_ids, unique_image_ids, sims)
    metrics["bidirectional_r1_mean"] = 0.5 * (metrics["r1_i2t"] + metrics["r1_t2i"])
    after = parameter_fingerprints(model)
    assert_unchanged(before, after)
    assert observed_shape[-2:] == (expected_resolution, expected_resolution), observed_shape
    return {
        "run_id": run_id,
        "observed_shape": observed_shape,
        "patch_size": patch_size,
        "patch_grid": f"{expected_grid}x{expected_grid}",
        "positional_interpolation": expected_interp,
        "parameters_unchanged": True,
        "optimizer_created": False,
        "metrics": metrics,
    }


def smoke_model_construction(run_id, device):
    cfg = load_config(run_id)
    model = DualEncoder(cfg).to(device)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    return {
        "run_id": run_id,
        "vision_depth": model.vision_depth,
        "patch_size": model.patch_size,
        "patch_grid": f"{model.patch_grid_height}x{model.patch_grid_width}",
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_module_names": trainable,
    }


def smoke_train_val_checkpoint(device):
    cfg = load_config("minimal_projonly")
    cfg["training"]["epochs"] = 1
    with tempfile.TemporaryDirectory(prefix="clipb_smoke_") as tmp:
        cfg["logging"]["checkpoint_dir"] = tmp
        tokenizer = CLIPTokenizer.from_pretrained(cfg["model"]["image_model_name"])
        train_loader = create_image_text_dataloader(cfg, tokenizer, split="train")
        val_loader = create_image_text_dataloader(cfg, tokenizer, split="val")
        model = DualEncoder(cfg).to(device)
        criterion = build_loss(cfg)
        optimizer = create_clip_optimizer(model, cfg)
        scheduler = create_lr_scheduler(optimizer, cfg, len(train_loader) * cfg["training"]["epochs"])
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device, tokenizer)
        trainer.train_epoch(0)
        score = trainer.evaluate(0)
        assert Path(tmp, "validation_metrics.jsonl").exists()
        assert trainer.save_checkpoint(1)
        ckpt_path = Path(tmp, "best_model.pth")
        trainer.load_checkpoint(str(ckpt_path))

        checkpoint = torch.load(ckpt_path, map_location=device)
        old = copy.deepcopy(trainer.config["data"]["image_size"])
        trainer.config["data"]["image_size"] = 336
        rejected = False
        try:
            trainer._validate_checkpoint_compatibility(checkpoint, ckpt_path)
        except ValueError:
            rejected = True
        finally:
            trainer.config["data"]["image_size"] = old
        assert rejected, "incompatible checkpoint was not rejected"
        return {
            "one_training_step_or_more": True,
            "validation_score": score,
            "checkpoint_saved": True,
            "checkpoint_reloaded": True,
            "incompatible_checkpoint_rejected": True,
        }


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paper_probe = load_config("minimal_uf4")
    thesis_probe = setup_config(
        config_path=CONFIG,
        overrides=load_registry_overrides("B0_uf4", registry_path=THESIS_REGISTRY),
    )
    assert paper_probe["model"]["image_model_name"] == "openai/clip-vit-base-patch16"
    assert thesis_probe["model"]["image_model_name"] == "openai/clip-vit-large-patch14-336"

    results = {
        "paper_registry_loading": True,
        "thesis_registry_loading": True,
        "zero_shot_336": smoke_zero_shot("zero_shot_336", 336, 21, True, device),
        "minimal_projonly_model": smoke_model_construction("minimal_projonly", device),
        "minimal_uf4_model": smoke_model_construction("minimal_uf4", device),
        "minimal_uf7_model": smoke_model_construction("minimal_uf7", device),
        "train_val_checkpoint": smoke_train_val_checkpoint(device),
        "unique_run_names": [
            "clipb16_zero-shot-r336_coco_s42",
            "clipb16_minimal-projonly-r336_coco_s42",
            "clipb16_minimal-uf1-r336_coco_s42",
            "clipb16_minimal-uf2-r336_coco_s42",
            "clipb16_minimal-uf3-r336_coco_s42",
            "clipb16_minimal-uf4-r336_coco_s42",
            "clipb16_minimal-uf5-r336_coco_s42",
            "clipb16_minimal-uf6-r336_coco_s42",
            "clipb16_minimal-uf7-r336_coco_s42",
        ],
    }
    assert len(results["unique_run_names"]) == len(set(results["unique_run_names"]))
    print(json.dumps(results, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
