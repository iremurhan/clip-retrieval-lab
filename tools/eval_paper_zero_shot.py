"""
Paper-specific zero-shot CLIP baseline calibration.

Evaluates a raw pretrained HuggingFace CLIP checkpoint on the repository's
Karpathy retrieval split. No DualEncoder wrapper, optimizer, checkpoint load,
checkpoint save, or parameter update is used.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import create_image_text_dataloader
from src.metrics import compute_recall_at_k
from src.setup import backbone_slug, load_registry_overrides, setup_config, setup_seed
from src.utils import chunked_matmul
from tools.eval_zero_shot import add_coco_eccv_metrics, compute_standard_metrics, format_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Paper zero-shot CLIP retrieval eval.")
    p.add_argument("--config", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--log-wandb", action="store_true")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def tensor_sha256(tensor):
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def parameter_fingerprints(model):
    return {
        name: (tuple(param.shape), str(param.dtype), tensor_sha256(param))
        for name, param in model.named_parameters()
    }


def assert_unchanged(before, after):
    if before != after:
        changed = [name for name in before if before.get(name) != after.get(name)]
        raise RuntimeError(f"Zero-shot evaluation changed model parameters: {changed[:20]}")


def call_image_features(model, images, positional_interpolation):
    if positional_interpolation:
        try:
            return model.get_image_features(
                pixel_values=images,
                interpolate_pos_encoding=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Installed transformers CLIPModel.get_image_features does not support "
                "interpolate_pos_encoding. Cannot run the 336 resolution control safely."
            ) from exc
    return model.get_image_features(pixel_values=images)


class CLIPZeroShotAdapter(torch.nn.Module):
    def __init__(self, clip_model, positional_interpolation):
        super().__init__()
        self.clip_model = clip_model
        self.positional_interpolation = positional_interpolation

    def encode_image(self, images):
        feats = call_image_features(self.clip_model, images, self.positional_interpolation)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        feats = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(feats, dim=-1)


@torch.no_grad()
def encode_images(model, loader, device, positional_interpolation):
    seen_ids = set()
    embeddings = []
    unique_ids = []
    observed_shape = None

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if observed_shape is None:
            observed_shape = tuple(images.shape)
        feats = call_image_features(model, images, positional_interpolation)
        feats = F.normalize(feats, dim=-1)

        ids = batch["image_id"]
        for i, img_id in enumerate(ids.tolist() if torch.is_tensor(ids) else ids):
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)
            embeddings.append(feats[i].detach().cpu())
            unique_ids.append(img_id)

    if not embeddings:
        raise RuntimeError("Loader yielded zero images; cannot evaluate.")
    return (
        torch.stack(embeddings, dim=0),
        torch.tensor(unique_ids, dtype=torch.long),
        observed_shape,
    )


@torch.no_grad()
def encode_texts(model, loader, device):
    embeddings = []
    image_ids = []
    sentids = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.append(F.normalize(feats, dim=-1).detach().cpu())
        ids = batch["image_id"]
        image_ids.extend(ids.tolist() if torch.is_tensor(ids) else ids)
        sid = batch["sentid"]
        sentids.extend(sid.tolist() if torch.is_tensor(sid) else sid)
    return (
        torch.cat(embeddings, dim=0),
        torch.tensor(image_ids, dtype=torch.long),
        torch.tensor(sentids, dtype=torch.long),
    )


def log_to_wandb(args, config, payload):
    import wandb

    project = args.wandb_project or config.get("logging", {}).get("wandb_project")
    run_name = args.wandb_run_name or config.get("logging", {}).get("wandb_run_name")
    group = args.wandb_group or config.get("logging", {}).get("wandb_group")
    wandb.init(
        project=project,
        name=run_name,
        group=group,
        config=payload["wandb_config"],
        tags=[],
        resume="allow",
    )
    log_data = {}
    split = payload["split"]
    for key, value in payload["metrics"].items():
        log_data[f"{split}/{key}"] = value
    log_data[f"{split}/bidirectional_r1_mean"] = payload["metrics"]["bidirectional_r1_mean"]
    wandb.log(log_data)
    for key, value in log_data.items():
        wandb.run.summary[key] = value
    wandb.run.summary["status"] = "completed"
    wandb.run.summary["selected_checkpoint_epoch"] = None
    wandb.finish()


def main():
    args = parse_args()
    setup_seed(args.seed)

    registry_overrides = load_registry_overrides(args.run, registry_path=args.registry)
    config = setup_config(config_path=args.config, overrides=registry_overrides)
    config.setdefault("logging", {})["registry_file"] = args.registry
    config.setdefault("logging", {})["registry_id"] = args.run
    config.setdefault("training", {})["seed"] = args.seed
    config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    model_id = config["model"]["image_model_name"]
    image_resolution = int(config["data"]["image_size"])
    positional_interpolation = bool(config["model"].get("positional_interpolation", False))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("Loading raw pretrained CLIP: %s", model_id)
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).float().to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    native_image_resolution = int(model.config.vision_config.image_size)
    patch_size = int(model.config.vision_config.patch_size)
    vision_depth = len(model.vision_model.encoder.layers)
    patch_grid = image_resolution // patch_size
    if image_resolution % patch_size != 0:
        raise ValueError(f"image_resolution={image_resolution} is not divisible by patch_size={patch_size}.")
    if image_resolution != native_image_resolution and not positional_interpolation:
        raise ValueError(
            f"Non-native image_resolution={image_resolution} requested for native "
            f"{native_image_resolution} without positional interpolation."
        )

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    loader = create_image_text_dataloader(config, tokenizer, split=args.split)
    before = parameter_fingerprints(model)

    logger.info("Encoding images on %s split ...", args.split)
    img_embeds, unique_image_ids, observed_shape = encode_images(
        model, loader, device, positional_interpolation
    )
    if observed_shape[-2:] != (image_resolution, image_resolution):
        raise RuntimeError(
            f"Observed image tensor spatial shape {observed_shape[-2:]}, expected "
            f"{image_resolution}x{image_resolution}."
        )
    logger.info("Encoding captions ...")
    txt_embeds, image_ids, sentids = encode_texts(model, loader, device)

    sims = chunked_matmul(img_embeds, txt_embeds)
    metrics = compute_standard_metrics(img_embeds, txt_embeds, image_ids, unique_image_ids, sims)
    metrics["bidirectional_r1_mean"] = 0.5 * (metrics["r1_i2t"] + metrics["r1_t2i"])
    if config["data"]["dataset"] == "coco" and args.split == "test" and not config.get("debug", {}).get("debug_mode", False):
        add_coco_eccv_metrics(metrics, sims, sentids, unique_image_ids)

    after = parameter_fingerprints(model)
    assert_unchanged(before, after)

    slug = backbone_slug(model_id)
    wandb_config = {
        "registry_file": args.registry,
        "registry_id": args.run,
        "display_name": config.get("logging", {}).get("display_name"),
        "experiment_family": config.get("logging", {}).get("experiment_family"),
        "resolution_role": config.get("logging", {}).get("resolution_role"),
        "backbone_model_id": model_id,
        "backbone_slug": slug,
        "dataset": config["data"]["dataset"],
        "seed": args.seed,
        "num_seeds": 1,
        "zero_shot": True,
        "evaluation_split": args.split,
        "image_resolution": image_resolution,
        "native_image_resolution": native_image_resolution,
        "positional_interpolation": positional_interpolation,
        "unfrozen_vision_blocks": 0,
        "vision_depth": vision_depth,
        "patch_size": patch_size,
        "patch_grid_height": patch_grid,
        "patch_grid_width": patch_grid,
        "vision_hidden_size": int(model.config.vision_config.hidden_size),
        "text_hidden_size": int(model.config.text_config.hidden_size),
        "native_projection_dimension": int(model.config.projection_dim),
        "trainable_parameter_count": 0,
        "trainable_parameter_percentage": 0.0,
        "selected_checkpoint_epoch": None,
        "git_commit": git_commit(),
    }
    payload = OrderedDict([
        ("status", "completed"),
        ("dataset", config["data"]["dataset"]),
        ("split", args.split),
        ("registry_file", args.registry),
        ("registry_id", args.run),
        ("model", model_id),
        ("backbone_slug", slug),
        ("image_resolution", image_resolution),
        ("native_image_resolution", native_image_resolution),
        ("positional_interpolation", positional_interpolation),
        ("observed_image_tensor_shape", observed_shape),
        ("patch_size", patch_size),
        ("patch_grid_height", patch_grid),
        ("patch_grid_width", patch_grid),
        ("vision_depth", vision_depth),
        ("trainable_parameter_count", 0),
        ("trainable_parameter_percentage", 0.0),
        ("optimizer_created", False),
        ("optimizer_steps", 0),
        ("parameters_unchanged", True),
        ("metrics", metrics),
        ("wandb_config", wandb_config),
    ])

    print("\n" + format_table(config["data"]["dataset"], f"zero_shot_r{image_resolution}", metrics) + "\n")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s", args.output)
    if args.log_wandb:
        log_to_wandb(args, config, payload)


if __name__ == "__main__":
    main()
