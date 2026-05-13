"""
Compute Wang & Isola alignment/uniformity diagnostics for retrieval checkpoints.

This script evaluates an existing fine-tuned checkpoint on the Karpathy test
split, using the same model loading and dataset conventions as the main
retrieval/SugarCrepe evaluation paths. It writes one JSON record per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.eval.eval_sugarcrepe import load_model_from_checkpoint
from src.data import create_image_text_dataloader


logger = logging.getLogger(__name__)


DATASET_ALIASES = {
    "coco": "coco",
    "flickr": "flickr30k",
    "flickr30k": "flickr30k",
}


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def parse_checkpoint_identity(checkpoint_path: str, config: dict) -> dict:
    """Derive run_id, dataset, and seed from checkpoint path/config."""
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

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    return {
        "run_id": path.parent.parent.name if path.parent.name.startswith("B") else path.parent.name,
        "dataset": DATASET_ALIASES.get(str(data_cfg.get("dataset", "unknown")), str(data_cfg.get("dataset", "unknown"))),
        "seed": int(train_cfg.get("seed", -1)),
    }


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_data_paths(config: dict, dataset: str, data_root: str) -> dict:
    """Return a config copy with dataset paths pointed at the local data root."""
    root = Path(data_root).expanduser()
    data_cfg = dict(config.get("data", {}))
    data_cfg["dataset"] = dataset

    if dataset == "coco":
        if root.name in {"val2014", "train2014", "val2017"}:
            coco_root = root.parent
        else:
            coco_root = root / "coco" if (root / "coco").exists() else root
        captions = _first_existing(
            [
                coco_root / "caption_datasets" / "dataset_coco.json",
                root / "caption_datasets" / "dataset_coco.json",
                Path(data_cfg.get("captions_path", "")),
            ]
        )
        images = _first_existing(
            [
                coco_root,
                root,
                Path(data_cfg.get("images_path", "")),
            ]
        )
        seg_map_dir = _first_existing([coco_root / "sam_masks", root / "sam_masks"])
        sam_feature_dir = _first_existing([coco_root / "sam_encoder_features", root / "sam_encoder_features"])
    elif dataset == "flickr30k":
        if root.name == "flickr30k_images":
            flickr_root = root.parent
        else:
            flickr_root = root / "flickr30k" if (root / "flickr30k").exists() else root
        captions = _first_existing(
            [
                flickr_root / "caption_datasets" / "dataset_flickr30k.json",
                root / "caption_datasets" / "dataset_flickr30k.json",
                Path(data_cfg.get("captions_path", "")),
            ]
        )
        images = _first_existing(
            [
                flickr_root / "flickr30k_images",
                root / "flickr30k_images",
                flickr_root,
                root,
                Path(data_cfg.get("images_path", "")),
            ]
        )
        seg_map_dir = _first_existing([flickr_root / "sam_masks", root / "sam_masks"])
        sam_feature_dir = _first_existing([flickr_root / "sam_encoder_features", root / "sam_encoder_features"])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if captions is None:
        raise FileNotFoundError(f"Could not find Karpathy captions JSON under {root}")
    if images is None:
        raise FileNotFoundError(f"Could not find images root under {root}")

    data_cfg["captions_path"] = str(captions)
    data_cfg["images_path"] = str(images)
    if seg_map_dir is not None:
        data_cfg["seg_map_dir"] = str(seg_map_dir)
    if sam_feature_dir is not None:
        data_cfg["sam_feature_dir"] = str(sam_feature_dir)

    out = dict(config)
    out["data"] = data_cfg
    out["debug"] = dict(out.get("debug", {}))
    out["debug"]["debug_mode"] = False
    return out


def build_tokenizer(config: dict):
    text_encoder_type = config.get("model", {}).get("text_encoder")
    if text_encoder_type == "blip":
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(config["model"]["text_model_name"])
    return CLIPTokenizer.from_pretrained(config["model"]["image_model_name"])


def maybe_cpu_fallback_for_mps(model, config: dict, device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    if device.type != "mps":
        return model, device
    try:
        image_size = int(config["data"]["image_size"])
        max_length = int(config["data"]["max_length"])
        dummy_img = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_ids = torch.zeros(1, max(5, min(max_length, 8)), dtype=torch.long, device=device)
        dummy_mask = torch.ones_like(dummy_ids)
        with torch.no_grad():
            model.encode_image(dummy_img)
            model.encode_text(dummy_ids, dummy_mask)
        logger.info("MPS smoke test passed.")
    except Exception as exc:
        logger.error("MPS smoke test failed (%s). Falling back to CPU.", exc)
        device = torch.device("cpu")
        model = model.to(device)
    return model, device


@torch.no_grad()
def extract_embeddings_and_alignment(model, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Extract unique image embeddings, all text embeddings, and cross-modal alignment."""
    model.eval()
    image_embeds_unique: list[torch.Tensor] = []
    text_embeds: list[torch.Tensor] = []
    seen_image_ids: set[int] = set()
    alignment_sum = 0.0
    alignment_count = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        image_ids = batch["image_id"].cpu().tolist()

        first_indices: list[int] = []
        local_lookup: dict[int, int] = {}
        inverse: list[int] = []
        for idx, image_id in enumerate(image_ids):
            if image_id not in local_lookup:
                local_lookup[image_id] = len(first_indices)
                first_indices.append(idx)
            inverse.append(local_lookup[image_id])

        first_idx_tensor = torch.tensor(first_indices, dtype=torch.long, device=images.device)
        unique_images = images.index_select(0, first_idx_tensor)

        seg_ids = None
        if "seg_ids" in batch:
            seg_ids = batch["seg_ids"].to(device, non_blocking=True).index_select(0, first_idx_tensor)
        sam_features = None
        if "sam_features" in batch:
            sam_features = batch["sam_features"].to(device, non_blocking=True).index_select(0, first_idx_tensor)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if sam_features is not None:
                img_unique = model.encode_image(unique_images, sam_features=sam_features)
            elif seg_ids is not None:
                img_unique = model.encode_image(unique_images, seg_ids=seg_ids)
            else:
                img_unique = model.encode_image(unique_images)
            txt = model.encode_text(input_ids, attention_mask)

        img_unique = F.normalize(img_unique.float(), p=2, dim=1)
        txt = F.normalize(txt.float(), p=2, dim=1)

        inverse_tensor = torch.tensor(inverse, dtype=torch.long, device=img_unique.device)
        img_for_caption = img_unique.index_select(0, inverse_tensor)
        sq_dist = (img_for_caption - txt).pow(2).sum(dim=1)
        alignment_sum += float(sq_dist.sum().item())
        alignment_count += int(sq_dist.numel())

        for local_pos, image_id in enumerate(local_lookup.keys()):
            if image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
                image_embeds_unique.append(img_unique[local_pos].detach().cpu())
        text_embeds.append(txt.detach().cpu())

        if step % 25 == 0:
            logger.info("Encoded %d batches: %d unique images, %d captions", step, len(seen_image_ids), alignment_count)

    if alignment_count == 0:
        raise RuntimeError("No image-caption pairs were encoded.")

    return torch.stack(image_embeds_unique, dim=0), torch.cat(text_embeds, dim=0), alignment_sum / alignment_count


def uniformity_full(embeddings: torch.Tensor, t: float, chunk_size: int = 512) -> float:
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    n = embeddings.shape[0]
    total = 0.0
    count = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = embeddings[start:end] @ embeddings.T
        d2 = (2.0 - 2.0 * sims).clamp_min_(0.0)
        vals = torch.exp(-t * d2)
        row_indices = torch.arange(start, end)
        vals[torch.arange(end - start), row_indices] = 0.0
        total += float(vals.sum().item())
        count += (end - start) * (n - 1)
    return math.log(total / count)


def uniformity_sampled(
    embeddings: torch.Tensor,
    t: float,
    num_pairs: int,
    seed: int,
    chunk_pairs: int = 250_000,
) -> float:
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError("Uniformity requires at least two embeddings.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    total = 0.0
    count = 0
    remaining = int(num_pairs)
    while remaining > 0:
        size = min(chunk_pairs, remaining)
        i = torch.randint(0, n, (size,), generator=gen)
        j = torch.randint(0, n - 1, (size,), generator=gen)
        j = j + (j >= i).long()
        dots = (embeddings[i] * embeddings[j]).sum(dim=1)
        d2 = (2.0 - 2.0 * dots).clamp_min_(0.0)
        total += float(torch.exp(-t * d2).sum().item())
        count += size
        remaining -= size
    return math.log(total / count)


def compute_uniformity(
    embeddings: torch.Tensor,
    t: float,
    num_pairs: int,
    seed: int,
    full_threshold: int,
) -> float:
    if embeddings.shape[0] <= full_threshold:
        logger.info("Computing exact uniformity over %d embeddings.", embeddings.shape[0])
        return uniformity_full(embeddings, t=t)
    logger.info(
        "Computing sampled uniformity over %d embeddings (%d pairs).",
        embeddings.shape[0],
        num_pairs,
    )
    return uniformity_sampled(embeddings, t=t, num_pairs=num_pairs, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alignment/uniformity diagnostic for CLIP retrieval checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["coco", "flickr30k"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--uniformity-pairs", type=int, default=2_000_000)
    parser.add_argument("--full-uniformity-threshold", type=int, default=6000)
    parser.add_argument("--uniformity-t", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = select_device(args.device)
    logger.info("Device: %s", device)

    model, checkpoint_config = load_model_from_checkpoint(args.checkpoint, device)
    identity = parse_checkpoint_identity(args.checkpoint, checkpoint_config)
    dataset = args.dataset or identity["dataset"]
    config = resolve_data_paths(checkpoint_config, dataset, args.data_root)
    config.setdefault("training", {})["batch_size"] = args.batch_size
    config.setdefault("data", {})["num_workers"] = args.num_workers

    model, device = maybe_cpu_fallback_for_mps(model, config, device)
    tokenizer = build_tokenizer(config)
    loader = create_image_text_dataloader(config, tokenizer, split="test")

    image_embs, text_embs, alignment = extract_embeddings_and_alignment(model, loader, device)
    logger.info("Final embedding shapes: images=%s, text=%s", tuple(image_embs.shape), tuple(text_embs.shape))

    seed = int(identity["seed"]) if identity["seed"] is not None else 0
    uniformity_img = compute_uniformity(
        image_embs,
        t=args.uniformity_t,
        num_pairs=args.uniformity_pairs,
        seed=seed + 17,
        full_threshold=args.full_uniformity_threshold,
    )
    uniformity_txt = compute_uniformity(
        text_embs,
        t=args.uniformity_t,
        num_pairs=args.uniformity_pairs,
        seed=seed + 31,
        full_threshold=args.full_uniformity_threshold,
    )

    result = {
        "run_id": identity["run_id"],
        "dataset": dataset,
        "seed": seed,
        "checkpoint": args.checkpoint,
        "n_images": int(image_embs.shape[0]),
        "n_captions": int(text_embs.shape[0]),
        "alignment": float(alignment),
        "uniformity_img": float(uniformity_img),
        "uniformity_txt": float(uniformity_txt),
        "uniformity_mean": float((uniformity_img + uniformity_txt) / 2.0),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Wrote %s", output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
