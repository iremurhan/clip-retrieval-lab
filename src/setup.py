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
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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

    Supports an optional `parent:` field per entry: parents are resolved
    recursively, depth-first, and the resulting kv pairs are concatenated
    parent-first → child-last so that child overrides win on conflict.
    Cycles raise ValueError; missing parents raise KeyError.

    Returns a list of 'key=value' strings compatible with apply_overrides().
    """
    if not os.path.isfile(registry_path):
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)
    runs = registry.get("runs", {})
    if run_id not in runs:
        raise KeyError(f"Run '{run_id}' not found in registry. Available: {list(runs.keys())}")

    def _collect(name, seen):
        if name in seen:
            raise ValueError(
                f"Registry parent cycle detected: {' -> '.join(list(seen) + [name])}"
            )
        if name not in runs:
            raise KeyError(
                f"Registry parent '{name}' (referenced by '{run_id}') not found. "
                f"Available: {list(runs.keys())}"
            )
        seen = seen | {name}
        entry = runs[name]
        result = []
        parent = entry.get("parent")
        if parent:
            result.extend(_collect(parent, seen))
        own = entry.get("overrides") or {}
        result.extend(f"{k}={v}" for k, v in own.items())
        return result

    return _collect(run_id, set())


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
    """Build a minimal config dict for WandB (scientific params only).

    Optionally forwards `logging.lineage` (a dict of provenance tags such as
    parent_branch / augmentation_scheme / previous_baseline) as both flat
    `lineage.*` keys and a nested `lineage` dict so WandB can group/filter
    by ancestry.
    """
    aug = config.get('augment', {})
    payload = {
        "run_id":                config['logging']['run_id'],
        "dataset":               config['data']['dataset'],
        "seed":                  config['training']['seed'],
        "model":                 config.get('model', {}).get('image_model_name'),
        "loss_type":             config.get('loss', {}).get('type', 'infonce'),
        "hard_negatives":        config.get('loss', {}).get('hard_negatives', False),
        "unfreeze_layers":       config.get('model', {}).get('unfreeze_vision_layers', 0),
        "intra_img_weight":      config.get('loss', {}).get('intra_img_weight', 0),
        "intra_txt_weight":      config.get('loss', {}).get('intra_txt_weight', 0),
        "k_photometric_augs":    aug.get('k_photometric_augs', 0),
        "aug_crop_scale_min":    aug.get('aug_crop_scale_min'),
        "color_jitter_strength": aug.get('color_jitter_strength'),
        "use_grayscale":         aug.get('use_grayscale'),
        "batch_size":            config.get('training', {}).get('batch_size'),
        "epochs":                config.get('training', {}).get('epochs'),
        "use_grad_cache":        config.get('training', {}).get('use_grad_cache', False),
    }

    lineage = config.get('logging', {}).get('lineage')
    if lineage is not None:
        if not isinstance(lineage, dict):
            raise TypeError(
                f"config['logging']['lineage'] must be a dict, got {type(lineage).__name__}"
            )
        for k, v in lineage.items():
            payload[f"lineage.{k}"] = v
        payload["lineage"] = dict(lineage)

    return payload


def format_run_name(run_id, dataset_name, seed=None):
    """Generate WandB run name: {run_id}_{dataset}_{seed}"""
    parts = [str(run_id), dataset_name]
    if seed is not None:
        parts.append(f"s{seed}")
    return "_".join(parts)


def setup_tracker(config):
    """
    Initialize WandB with clean config and run name.
    run_id from config['logging']['run_id']; dataset and seed from config.
    """
    if not config.get("logging", {}).get("use_wandb", True):
        return
    project = config.get("logging", {}).get("wandb_project")
    if not project:
        logger.warning("wandb_project not set; skipping WandB init.")
        return

    run_id = config['logging']['run_id']
    seed = config['training']['seed']
    dataset_name = config['data']['dataset']

    run_name = format_run_name(run_id, dataset_name, seed=seed)
    group = run_id

    tags = [
        f"run_id:{run_id}",
        f"dataset:{dataset_name}",
        f"seed:{seed}",
    ]
    # Forward each logging.lineage entry as a filterable WandB tag so that
    # default-bumps (e.g. augmentation magnitudes v1 → v2) split cleanly in
    # the runs page without needing a config-key query.
    lineage = config.get("logging", {}).get("lineage")
    if isinstance(lineage, dict):
        for k, v in lineage.items():
            tags.append(f"lineage.{k}:{v}")

    wandb_id = config.get("logging", {}).get("wandb_id")

    init_kwargs = {
        "project": project,
        "config": make_wandb_config(config),
        "name": run_name,
        "group": group,
        "tags": tags,
    }

    # If a specific WandB run id is provided, force resume onto that run
    if wandb_id:
        init_kwargs["id"] = wandb_id
        init_kwargs["resume"] = "must"
    else:
        # Otherwise allow WandB to create a new run or resume heuristically
        init_kwargs["resume"] = "allow"

    try:
        wandb.init(**init_kwargs)
    except Exception as e:
        logger.error(
            f"WandB init failed (non-fatal): {e}. "
            "Training will proceed without WandB logging."
        )
        return

    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    wandb.config.update(
        {'slurm_job_id': slurm_job_id},
        allow_val_change=True
    )
    logger.info(f"WandB run: {run_name} | group: {group} | slurm_job_id: {slurm_job_id}")
