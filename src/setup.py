"""
src/setup.py
------------
Configuration and setup utilities for the Cross-Modal Retrieval project.
Handles config loading, merging, CLI overrides, seed setup, and WandB tracker initialization.
"""

import os
import yaml
import torch
import numpy as np
import random
import logging
import wandb

logger = logging.getLogger(__name__)


def setup_seed(seed=42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def deep_merge_dicts(base, override):
    """Recursively merge override dict into base (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_merge_dicts(base[k], v)
        else:
            base[k] = v


def apply_overrides(config, overrides):
    """Apply CLI overrides (list of 'key=value') into nested config."""
    if not overrides:
        return
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.strip().split(".")
        current = config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        # Type casting
        raw = value.strip()
        if raw.lower() == "true":
            raw = True
        elif raw.lower() == "false":
            raw = False
        else:
            try:
                if "." in raw:
                    raw = float(raw)
                else:
                    raw = int(raw)
            except ValueError:
                pass
        current[keys[-1]] = raw


def setup_config(base_path=None, config_path=None, overrides=None):
    """
    Load base config, optionally merge dataset-specific config, apply CLI overrides.
    If only config_path is given, base is taken as config_base.yaml in the same directory.
    Returns the merged config dict. Does not create directories (handled by SLURM/run script).
    """
    if base_path is None and config_path:
        base_path = os.path.join(os.path.dirname(config_path), "config_base.yaml")
    if not base_path or not os.path.isfile(base_path):
        raise FileNotFoundError(f"Base config not found: {base_path}")
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            specific = yaml.safe_load(f)
        deep_merge_dicts(config, specific)
    if overrides:
        apply_overrides(config, overrides)
    return config


def make_wandb_config(config):
    """Build a minimal config dict for WandB (scientific params only)."""
    images_path = config.get("data", {}).get("images_path", "")
    dataset_name = "flickr30k" if "flickr" in images_path else "coco"
    return {
        "dataset": dataset_name,
        "model": {
            "name": config.get("model", {}).get("image_model_name"),
            "unfreeze": config.get("model", {}).get("unfreeze_vision_layers"),
            "strategy": config.get("model", {}).get("unfreeze_strategy"),
        },
        "batch_size": config.get("training", {}).get("batch_size"),
        "max_length": config.get("data", {}).get("max_length"),
        "num_workers": config.get("data", {}).get("num_workers"),
        "image_size": config.get("data", {}).get("image_size"),
        "epochs": config.get("training", {}).get("epochs"),
        "lr": {
            "head": config.get("training", {}).get("head_lr"),
            "backbone": config.get("training", {}).get("backbone_lr"),
            "clip": config.get("training", {}).get("clip_projection_lr"),
        },
    }


def format_run_name(job_id, dataset_name, exp_name=""):
    """Generate WandB run name: {job_id}_{dataset_name}_{exp_name} or {job_id}_{dataset_name}."""
    parts = [str(job_id), dataset_name]
    if exp_name:
        parts.append(exp_name)
    return "_".join(parts)


def setup_tracker(config, debug_mode=False):
    """
    Initialize WandB with clean config and run name.
    job_id from SLURM_JOB_ID or 'local'; dataset from config; exp_name optional (e.g. from config).
    """
    if not config.get("logging", {}).get("use_wandb", True) or debug_mode:
        return
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    images_path = config.get("data", {}).get("images_path", "")
    dataset_name = "flickr30k" if "flickr" in images_path else "coco"
    exp_name = config.get("logging", {}).get("run_name", "") or ""
    run_name = format_run_name(job_id, dataset_name, exp_name)
    project = config.get("logging", {}).get("wandb_project")
    if not project:
        logger.warning("wandb_project not set; skipping WandB init.")
        return

    wandb_id = config.get("logging", {}).get("wandb_id")
    
    # Read group and job_type from config (support usage under 'wandb' key or 'logging')
    group = config.get("wandb", {}).get("group") or config.get("logging", {}).get("group")
    job_type = config.get("wandb", {}).get("job_type") or config.get("logging", {}).get("job_type")

    init_kwargs = {
        "project": project,
        "config": make_wandb_config(config),
        "name": run_name,
    }
    
    if group:
        init_kwargs["group"] = group
    if job_type:
        init_kwargs["job_type"] = job_type

    # If a specific WandB run id is provided, force resume onto that run
    if wandb_id:
        init_kwargs["id"] = wandb_id
        init_kwargs["resume"] = "must"
    else:
        # Otherwise allow WandB to create a new run or resume heuristically
        init_kwargs["resume"] = "allow"

    wandb.init(**init_kwargs)
