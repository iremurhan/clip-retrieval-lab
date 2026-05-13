"""
Patch-level SugarCrepe diagnostic.

This evaluates whether compositional failures are caused by CLS pooling by
comparing the standard image embedding against a patch-max image score:

  CLS:       sim(encode_image(image), caption)
  Patch-max: max_patch sim(encode_image_patches(image)[patch], caption)

BLIP_TEXT checkpoints are supported because they share the CLIP vision tower;
only the text tokenizer/encoder changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.eval.eval_sugarcrepe import load_model_from_checkpoint
from src.data import build_eval_transform
from src.eval.sugarcrepe import SUBCATEGORIES


logger = logging.getLogger(__name__)

DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def parse_checkpoint_identity(checkpoint_path: str, config: dict) -> dict:
    path = Path(checkpoint_path)
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)$")
    for part in reversed(path.parts):
        match = pattern.match(part)
        if match:
            return {
                "run_id": match.group("run"),
                "dataset": DATASET_ALIASES[match.group("dataset")],
                "seed": int(match.group("seed")),
            }
    return {
        "run_id": path.parent.parent.name if path.parent.name.startswith("B") else path.parent.name,
        "dataset": DATASET_ALIASES.get(str(config.get("data", {}).get("dataset", "unknown")), "unknown"),
        "seed": int(config.get("training", {}).get("seed", -1)),
    }


def build_tokenizer(config: dict):
    if config.get("model", {}).get("text_encoder") == "blip":
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(config["model"]["text_model_name"])
    return CLIPTokenizer.from_pretrained(config["model"]["image_model_name"])


def resolve_image_path(images_dir: str, filename: str) -> str | None:
    direct = os.path.join(images_dir, filename)
    if os.path.isfile(direct):
        return direct
    prefixed_paths = [
        os.path.join(images_dir, f"COCO_val2017_{filename}"),
        os.path.join(images_dir, f"COCO_val2014_{filename}"),
    ]
    return next((p for p in prefixed_paths if os.path.isfile(p)), None)


def load_subcategory(data_dir: str, subcategory: str) -> dict:
    json_path = os.path.join(data_dir, f"{subcategory}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"SugarCrepe data file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def run_patch_sanity_check(model, tokenizer, transform, data_dir: str, images_dir: str, max_length: int, device: torch.device) -> None:
    first_entry = None
    for subcat in SUBCATEGORIES:
        data = load_subcategory(data_dir, subcat)
        if data:
            first_entry = next(iter(data.values()))
            break
    if first_entry is None:
        raise RuntimeError("Could not find any SugarCrepe item for patch sanity check.")

    img_path = resolve_image_path(images_dir, first_entry["filename"])
    if img_path is None:
        raise FileNotFoundError(f"Sanity-check image not found under {images_dir}: {first_entry['filename']}")

    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    patches = model.encode_image_patches(img_tensor)
    if patches.ndim != 3 or patches.size(0) != 1:
        raise RuntimeError(f"encode_image_patches returned unexpected shape: {tuple(patches.shape)}")
    if not torch.isfinite(patches).all():
        raise RuntimeError("encode_image_patches returned NaN/Inf values.")

    tokenized = tokenizer(
        first_entry["caption"],
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    text_emb = model.encode_text(tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device))
    if patches.size(-1) != text_emb.size(-1):
        raise RuntimeError(
            f"Patch/text embedding dim mismatch: patches={patches.size(-1)}, text={text_emb.size(-1)}"
        )
    patch_norms = patches.norm(dim=-1)
    logger.info(
        "Patch sanity check passed: shape=%s, norm mean=%.4f, norm std=%.4f",
        tuple(patches.shape),
        float(patch_norms.mean().item()),
        float(patch_norms.std().item()),
    )


@torch.no_grad()
def evaluate_subcategory_patch_level(
    model,
    tokenizer,
    transform,
    data: dict,
    images_dir: str,
    max_length: int,
    device: torch.device,
) -> dict:
    cls_correct = 0
    patch_correct = 0
    total = 0

    for entry in data.values():
        img_path = resolve_image_path(images_dir, entry["filename"])
        if img_path is None:
            logger.warning("Image not found, skipping: %s", os.path.join(images_dir, entry["filename"]))
            continue

        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        tok_pos = tokenizer(
            entry["caption"],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        tok_neg = tokenizer(
            entry["negative_caption"],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        pos_ids = tok_pos["input_ids"].to(device)
        pos_mask = tok_pos["attention_mask"].to(device)
        neg_ids = tok_neg["input_ids"].to(device)
        neg_mask = tok_neg["attention_mask"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            img_cls = model.encode_image(img_tensor)
            img_patches = model.encode_image_patches(img_tensor)
            pos_emb = model.encode_text(pos_ids, pos_mask)
            neg_emb = model.encode_text(neg_ids, neg_mask)

        img_cls = F.normalize(img_cls.float(), p=2, dim=-1)
        img_patches = F.normalize(img_patches.float(), p=2, dim=-1)
        pos_emb = F.normalize(pos_emb.float(), p=2, dim=-1)
        neg_emb = F.normalize(neg_emb.float(), p=2, dim=-1)

        cls_pos = F.cosine_similarity(img_cls, pos_emb).item()
        cls_neg = F.cosine_similarity(img_cls, neg_emb).item()
        pos_patch_max = (img_patches[0] @ pos_emb[0]).max().item()
        neg_patch_max = (img_patches[0] @ neg_emb[0]).max().item()

        cls_correct += int(cls_pos > cls_neg)
        patch_correct += int(pos_patch_max > neg_patch_max)
        total += 1

    if total == 0:
        raise RuntimeError("No valid SugarCrepe samples found. Check images_dir and data files.")

    cls_acc = cls_correct / total
    patch_acc = patch_correct / total
    return {
        "cls_accuracy": cls_acc,
        "patch_max_accuracy": patch_acc,
        "delta": patch_acc - cls_acc,
        "n_items": total,
    }


def evaluate_patch_level(
    model,
    tokenizer,
    transform,
    device: torch.device,
    data_dir: str,
    images_dir: str,
    max_length: int,
    splits: tuple[str, ...],
) -> dict:
    selected = [s for s in SUBCATEGORIES if any(s.startswith(prefix) for prefix in splits)]
    if not selected:
        raise ValueError(f"No SugarCrepe subcategories match splits={splits}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    results = {}
    for subcat in selected:
        data = load_subcategory(data_dir, subcat)
        metrics = evaluate_subcategory_patch_level(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            data=data,
            images_dir=images_dir,
            max_length=max_length,
            device=device,
        )
        results[subcat] = metrics
        logger.info(
            "  %s: cls=%.4f patch_max=%.4f delta=%+.4f n=%d",
            subcat,
            metrics["cls_accuracy"],
            metrics["patch_max_accuracy"],
            metrics["delta"],
            metrics["n_items"],
        )

    results["macro_avg"] = {
        "cls": sum(results[s]["cls_accuracy"] for s in selected) / len(selected),
        "patch_max": sum(results[s]["patch_max_accuracy"] for s in selected) / len(selected),
    }
    results["macro_avg"]["delta"] = results["macro_avg"]["patch_max"] - results["macro_avg"]["cls"]
    results["macro_avg"]["n_items"] = sum(results[s]["n_items"] for s in selected)
    return results


def maybe_cpu_fallback_for_mps(model, config: dict, device: torch.device):
    if device.type != "mps":
        return model, device
    try:
        image_size = config["data"]["image_size"]
        dummy_img = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_ids = torch.zeros(1, 5, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1, 5, dtype=torch.long, device=device)
        with torch.no_grad():
            model.encode_image(dummy_img)
            model.encode_image_patches(dummy_img)
            model.encode_text(dummy_ids, dummy_mask)
        logger.info("MPS smoke test passed.")
    except Exception as exc:
        logger.error("MPS smoke test failed (%s). Falling back to CPU.", exc)
        device = torch.device("cpu")
        model = model.to(device)
    return model, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-level SugarCrepe diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = select_device(args.device)
    logger.info("Device: %s", device)

    model, config = load_model_from_checkpoint(args.checkpoint, device)
    model, device = maybe_cpu_fallback_for_mps(model, config, device)
    tokenizer = build_tokenizer(config)
    transform = build_eval_transform(config["data"]["image_size"])
    max_length = config["data"]["max_length"]

    run_patch_sanity_check(model, tokenizer, transform, args.data_dir, args.images_dir, max_length, device)

    identity = parse_checkpoint_identity(args.checkpoint, config)
    results = evaluate_patch_level(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device=device,
        data_dir=args.data_dir,
        images_dir=args.images_dir,
        max_length=max_length,
        splits=("replace", "swap", "add"),
    )

    payload = {
        **identity,
        "checkpoint": args.checkpoint,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Wrote %s", output)

    if args.wandb_run_id:
        import wandb

        project = args.wandb_project or config.get("logging", {}).get("wandb_project", "clip-retrieval")
        wandb.init(id=args.wandb_run_id, project=project, resume="must")
        for subcat, metrics in results.items():
            if subcat == "macro_avg":
                wandb.run.summary["sugarcrepe_patch/macro_avg/cls_acc"] = metrics["cls"]
                wandb.run.summary["sugarcrepe_patch/macro_avg/patch_max_acc"] = metrics["patch_max"]
                wandb.run.summary["sugarcrepe_patch/macro_avg/delta"] = metrics["delta"]
            else:
                wandb.run.summary[f"sugarcrepe_patch/{subcat}/cls_acc"] = metrics["cls_accuracy"]
                wandb.run.summary[f"sugarcrepe_patch/{subcat}/patch_max_acc"] = metrics["patch_max_accuracy"]
                wandb.run.summary[f"sugarcrepe_patch/{subcat}/delta"] = metrics["delta"]
        wandb.finish()
        logger.info("Logged patch-level results to WandB.")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
