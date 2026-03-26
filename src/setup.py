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
    torch.backends.cudnn.benchmark = False


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


def load_registry_overrides(run_id, registry_path="configs/registry.yaml"):
    """
    Load the overrides list for a named run from the registry.
    Returns a list of 'key=value' strings compatible with apply_overrides().
    Raises KeyError if run_id is not found in the registry.
    """
    if not os.path.isfile(registry_path):
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)
    runs = registry.get("runs", {})
    if run_id not in runs:
        raise KeyError(f"Run '{run_id}' not found in registry. Available: {list(runs.keys())}")
    overrides_dict = runs[run_id].get("overrides") or {}
    return [f"{k}={v}" for k, v in overrides_dict.items()]


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


_DATASET_KEYWORDS = {
    "flickr30k": "flickr",
    "coco": "coco",
}


def _infer_dataset_name(config):
    """
    Infer dataset name from data.images_path by keyword matching.
    Returns the matched dataset name, or 'unknown' if no keyword matches.
    """
    images_path = config.get("data", {}).get("images_path", "").lower()
    for name, keyword in _DATASET_KEYWORDS.items():
        if keyword in images_path:
            return name
    return "unknown"


_VALID_SEEDS = {42, 123, 456}


def make_wandb_config(config):
    """Build config dict for WandB (scientific params only)."""
    dataset_name = _infer_dataset_name(config)
    seed = config.get("training", {}).get("seed", 42)
    run_id = config.get("logging", {}).get("run_id", "unnamed")
    # wise_alpha is None for standard runs; a float only for WiSE-FT post-hoc runs.
    # Using None (not 0.0) because 0.0 is a valid interpolation weight.
    wise_alpha = config.get("logging", {}).get("wise_alpha", None)
    return {
        "dataset": dataset_name,
        "run_id": run_id,
        "seed": seed,
        "loss_type": config.get("loss", {}).get("type", "infonce"),
        "hard_negatives": config.get("loss", {}).get("hard_negatives", False),
        "lora_rank": config.get("model", {}).get("lora_rank", 0),
        "unfreeze_layers": config.get("model", {}).get("unfreeze_vision_layers"),
        "intra_img_weight": config.get("loss", {}).get("intra_img_weight", 0.0),
        "intra_txt_weight": config.get("loss", {}).get("intra_txt_weight", 0.0),
        "k_photometric_augs": config.get("augment", {}).get("k_photometric_augs"),
        "wise_alpha": wise_alpha,
        "model": {
            "name": config.get("model", {}).get("image_model_name"),
            "unfreeze": config.get("model", {}).get("unfreeze_vision_layers"),
            "strategy": config.get("model", {}).get("unfreeze_strategy"),
        },
        "batch_size": config.get("training", {}).get("batch_size"),
        "epochs": config.get("training", {}).get("epochs"),
        "lr": {
            "backbone": config.get("training", {}).get("backbone_lr"),
            "clip": config.get("training", {}).get("clip_projection_lr"),
        },
    }


def setup_tracker(config, debug_mode=False):
    """
    Initialize WandB with clean config and run name.

    Run naming:
      Standard : {run_id}_{dataset}_s{seed}   e.g. B0_coco_s42
      WiSE-FT  : {run_id}_{dataset}_wise{alpha} e.g. FULL_coco_wise0.5
                 (no seed suffix — WiSE-FT interpolation is deterministic)

    Group is set to run_id so all seeds of the same config are grouped together
    and WandB can show mean ± std in the group view.

    Valid seeds for standard runs: 42, 123, 456.
    """
    if not config.get("logging", {}).get("use_wandb", True) or debug_mode:
        return

    project = config.get("logging", {}).get("wandb_project")
    if not project:
        raise ValueError(
            "logging.wandb_project is not set but logging.use_wandb is true. "
            "Set wandb_project in config or pass it via --override, or set use_wandb: false."
        )

    dataset_name = _infer_dataset_name(config)
    run_id = config.get("logging", {}).get("run_id", "unnamed")
    seed = config.get("training", {}).get("seed", 42)
    wise_alpha = config.get("logging", {}).get("wise_alpha", None)

    is_wise = wise_alpha is not None

    if not is_wise:
        if seed not in _VALID_SEEDS:
            raise ValueError(
                f"seed={seed} is not a valid seed. "
                f"Valid seeds are: {sorted(_VALID_SEEDS)}. "
                "Use one of these or set logging.wise_alpha for a WiSE-FT run."
            )
        run_name = f"{run_id}_{dataset_name}_s{seed}"
    else:
        run_name = f"{run_id}_{dataset_name}_wise{wise_alpha}"

    wandb_id = config.get("logging", {}).get("wandb_id")
    job_type = config.get("wandb", {}).get("job_type") or config.get("logging", {}).get("job_type")

    tags = [
        f"dataset:{dataset_name}",
        f"run_id:{run_id}",
    ]
    if is_wise:
        tags.append(f"phase:wise")
        tags.append(f"alpha:{wise_alpha}")
    else:
        tags.append(f"phase:train")
        tags.append(f"seed:{seed}")

    init_kwargs = {
        "project": project,
        "config": make_wandb_config(config),
        "name": run_name,
        "group": run_id,
        "tags": tags,
    }

    if job_type:
        init_kwargs["job_type"] = job_type

    # If a specific WandB run id is provided, force resume onto that run
    if wandb_id:
        init_kwargs["id"] = wandb_id
        init_kwargs["resume"] = "must"
    else:
        init_kwargs["resume"] = "allow"

    wandb.init(**init_kwargs)
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    wandb.config.update({'slurm_job_id': slurm_job_id}, allow_val_change=True)
