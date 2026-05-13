from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from transformers import CLIPTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.eval.eval_cross_dataset import extract_embeddings, select_device
from scripts.eval.eval_sugarcrepe import load_model_from_checkpoint
from src.data import create_image_text_dataloader
from src.metrics import build_ranked_dicts
from src.utils import chunked_matmul


logger = logging.getLogger(__name__)
DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}


def build_tokenizer(config: dict):
    if config.get("model", {}).get("text_encoder") == "blip":
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(config["model"]["text_model_name"])
    return CLIPTokenizer.from_pretrained(config["model"]["image_model_name"])


def parse_identity(checkpoint_path: str, config: dict, wandb_run_name: str | None = None) -> dict:
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)$")
    for candidate in [wandb_run_name or "", *reversed(Path(checkpoint_path).parts)]:
        match = pattern.match(candidate)
        if match:
            return {
                "run_id": match.group("run"),
                "dataset": DATASET_ALIASES[match.group("dataset")],
                "seed": int(match.group("seed")),
            }
    return {
        "run_id": Path(checkpoint_path).parents[1].name,
        "dataset": DATASET_ALIASES.get(str(config.get("data", {}).get("dataset", "")), "unknown"),
        "seed": int(config.get("training", {}).get("seed", -1)),
    }


def coco_config(config: dict, data_root: str | Path, batch_size: int | None, num_workers: int | None) -> dict:
    cfg = json.loads(json.dumps(config))
    root = Path(data_root) / "coco"
    cfg.setdefault("data", {}).update(
        {
            "dataset": "coco",
            "images_path": str(root),
            "captions_path": str(root / "caption_datasets" / "dataset_coco.json"),
            "seg_map_dir": str(root / "sam_masks"),
            "sam_feature_dir": str(root / "sam_encoder_features"),
        }
    )
    cfg.setdefault("debug", {})["debug_mode"] = False
    if batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    if num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = int(num_workers)
    return cfg


def embedding_cache_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    return cache_dir / "embeddings" / f"{run_id.replace('/', '_')}_{dataset}_s{seed}_coco.pt"


def stats_cache_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    return cache_dir / "stats" / f"{run_id.replace('/', '_')}_{dataset}_s{seed}_coco.json"


def load_or_compute_embeddings(
    checkpoint_path: str,
    identity: dict,
    device: torch.device,
    data_root: str,
    cache_dir: Path,
    batch_size: int | None,
    num_workers: int,
    force_embeddings: bool,
) -> tuple:
    emb_path = embedding_cache_path(cache_dir, identity["run_id"], identity["dataset"], identity["seed"])
    if emb_path.exists() and not force_embeddings:
        logger.info("Loading embedding cache: %s", emb_path)
        payload = torch.load(emb_path, map_location="cpu", weights_only=False)
        return (
            payload["img_embeds_unique"],
            payload["txt_embeds"],
            payload["image_ids"],
            payload["unique_image_ids"],
            payload["sentids"],
            payload["unique_image_ids_list"],
        )

    model, config = load_model_from_checkpoint(checkpoint_path, device)
    tokenizer = build_tokenizer(config)
    loader = create_image_text_dataloader(
        coco_config(config, data_root, batch_size=batch_size, num_workers=num_workers),
        tokenizer,
        split="test",
    )
    embeddings = extract_embeddings(model, loader, device, use_amp=(device.type == "cuda"))
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "img_embeds_unique": embeddings[0],
            "txt_embeds": embeddings[1],
            "image_ids": embeddings[2],
            "unique_image_ids": embeddings[3],
            "sentids": embeddings[4],
            "unique_image_ids_list": embeddings[5],
        },
        emb_path,
    )
    logger.info("Wrote embedding cache: %s", emb_path)
    return embeddings


def _fraction(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def compute_direction_stats(ranked: dict, standard_gts: dict, eccv_gts: dict, topk: int = 5) -> tuple[float, float, int]:
    recovered_num = 0
    recovered_den = 0
    neigh_num = 0
    neigh_den = 0
    n_queries = 0
    for query_id, retrieved in ranked.items():
        if query_id not in standard_gts or query_id not in eccv_gts:
            continue
        n_queries += 1
        standard = set(standard_gts[query_id])
        extended = set(eccv_gts[query_id])
        if not retrieved:
            continue
        top1 = retrieved[0]
        if top1 not in standard:
            recovered_den += 1
            recovered_num += int(top1 in extended)
        for item in retrieved[:topk]:
            if item in standard:
                continue
            neigh_den += 1
            neigh_num += int(item in extended)
    return _fraction(recovered_num, recovered_den), _fraction(neigh_num, neigh_den), n_queries


def compute_missing_positive_stats(embeddings: tuple) -> dict:
    from eccv_caption import Metrics

    img_embeds_unique, txt_embeds, _image_ids, _unique_image_ids, sentids, unique_image_ids_list = embeddings
    sims = chunked_matmul(img_embeds_unique, txt_embeds)
    i2t_ranked, t2i_ranked = build_ranked_dicts(
        sims.t().numpy(),
        image_ids=[int(x) for x in unique_image_ids_list],
        caption_ids=[int(x) for x in sentids.tolist()],
    )
    metric = Metrics()
    i2t_top1, i2t_top5, n_i2t = compute_direction_stats(i2t_ranked, metric.coco_gts["i2t"], metric.eccv_gts["i2t"])
    t2i_top1, t2i_top5, n_t2i = compute_direction_stats(t2i_ranked, metric.coco_gts["t2i"], metric.eccv_gts["t2i"])
    return {
        "eccv_top1_recovered_i2t": i2t_top1 * 100.0,
        "eccv_top1_recovered_t2i": t2i_top1 * 100.0,
        "eccv_top5_neighborhood_i2t": i2t_top5 * 100.0,
        "eccv_top5_neighborhood_t2i": t2i_top5 * 100.0,
        "n_i2t_queries": n_i2t,
        "n_t2i_queries": n_t2i,
    }


def evaluate_checkpoint(entry: dict, args: argparse.Namespace, device: torch.device) -> Path | None:
    checkpoint = entry["checkpoint_path"]
    if not os.path.isfile(checkpoint):
        logger.error("Checkpoint not found: %s", checkpoint)
        return None
    model_config = torch.load(checkpoint, map_location="cpu", weights_only=False).get("config", {})
    identity = parse_identity(checkpoint, model_config, entry.get("wandb_run_name"))
    if identity["dataset"] != "coco" and not args.include_non_coco:
        logger.info("Skipping non-COCO checkpoint for COCO stats: %s", entry.get("wandb_run_name", checkpoint))
        return None
    out_path = stats_cache_path(Path(args.cache_dir), identity["run_id"], identity["dataset"], identity["seed"])
    if out_path.exists() and not args.force:
        logger.info("Stats cache exists, skipping: %s", out_path)
        return out_path
    embeddings = load_or_compute_embeddings(
        checkpoint,
        identity,
        device,
        args.data_root,
        Path(args.cache_dir),
        args.batch_size,
        args.num_workers,
        args.force_embeddings,
    )
    stats = compute_missing_positive_stats(embeddings)
    payload = {**identity, "checkpoint_path": checkpoint, "metrics": stats}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)
    for key, value in stats.items():
        if key.startswith("eccv_"):
            logger.info("  %s: %.4f", key, value)
    return out_path


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute COCO ECCV missing-positive diagnostics")
    parser.add_argument("--manifest", type=str, default="sugarcrepe_manifest.csv")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="/Volumes/T7/Research/experiments/datasets")
    parser.add_argument("--cache-dir", type=str, default="results/cache/missing_positive_stats")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--include-non-coco", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    device = select_device(args.device)
    if args.checkpoint:
        entries = [{"checkpoint_path": args.checkpoint, "wandb_run_name": args.wandb_run_name or ""}]
    else:
        entries = load_manifest(args.manifest)[args.start :]
        if args.limit is not None:
            entries = entries[: args.limit]
    for entry in entries:
        evaluate_checkpoint(entry, args, device)


if __name__ == "__main__":
    main()
