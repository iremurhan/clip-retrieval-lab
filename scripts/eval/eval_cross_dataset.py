"""
Evaluate a trained retrieval checkpoint on the opposite dataset test split.

No fine-tuning is performed. A Flickr30K-trained checkpoint is evaluated on
COCO Karpathy test, and a COCO-trained checkpoint is evaluated on Flickr30K
Karpathy test. Metrics are written to a per-checkpoint JSON cache and can be
logged to the source WandB run under the ood/* prefix.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from transformers import CLIPTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.eval.eval_sugarcrepe import load_model_from_checkpoint
from src.data import create_image_text_dataloader
from src.metrics import _build_gt_mappings, build_ranked_dicts, compute_eccv_metrics, compute_mapr_rprecision, compute_recall_at_k
from src.utils import chunked_matmul


logger = logging.getLogger(__name__)
DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}
OPPOSITE_DATASET = {"coco": "flickr30k", "flickr30k": "coco"}


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def build_tokenizer(config: dict):
    if config.get("model", {}).get("text_encoder") == "blip":
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(config["model"]["text_model_name"])
    return CLIPTokenizer.from_pretrained(config["model"]["image_model_name"])


def parse_identity(checkpoint_path: str, config: dict, wandb_run_name: str | None = None) -> dict:
    candidates = [wandb_run_name or "", *reversed(Path(checkpoint_path).parts)]
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)$")
    for candidate in candidates:
        match = pattern.match(candidate)
        if match:
            return {
                "run_id": match.group("run"),
                "train_dataset": DATASET_ALIASES[match.group("dataset")],
                "seed": int(match.group("seed")),
            }

    train_dataset = DATASET_ALIASES.get(str(config.get("data", {}).get("dataset", "")), "unknown")
    return {
        "run_id": Path(checkpoint_path).parents[1].name,
        "train_dataset": train_dataset,
        "seed": int(config.get("training", {}).get("seed", -1)),
    }


def dataset_paths(dataset: str, data_root: str | Path) -> dict:
    root = Path(data_root)
    if dataset == "coco":
        base = root / "coco"
        return {
            "dataset": "coco",
            "images_path": str(base),
            "captions_path": str(base / "caption_datasets" / "dataset_coco.json"),
            "seg_map_dir": str(base / "sam_masks"),
            "sam_feature_dir": str(base / "sam_encoder_features"),
        }
    if dataset == "flickr30k":
        base = root / "flickr30k"
        return {
            "dataset": "flickr30k",
            "images_path": str(base / "flickr30k_images"),
            "captions_path": str(base / "caption_datasets" / "dataset_flickr30k.json"),
            "seg_map_dir": str(base / "sam_masks"),
            "sam_feature_dir": str(base / "sam_encoder_features"),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_eval_config(
    checkpoint_config: dict,
    eval_dataset: str,
    data_root: str | Path,
    batch_size: int | None,
    num_workers: int | None,
) -> dict:
    config = copy.deepcopy(checkpoint_config)
    config.setdefault("data", {}).update(dataset_paths(eval_dataset, data_root))
    config.setdefault("debug", {})["debug_mode"] = False
    if batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(batch_size)
    if num_workers is not None:
        config.setdefault("data", {})["num_workers"] = int(num_workers)
    return config


@torch.no_grad()
def extract_embeddings(model, loader, device: torch.device, use_amp: bool) -> tuple:
    model.eval()
    img_embeds_list = []
    txt_embeds_list = []
    image_ids_list = []
    sentids_list = []

    use_seg_ids = loader.dataset.seg_loader is not None
    use_sam_features = loader.dataset.sam_feature_loader is not None

    for idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        seg_ids = batch["seg_ids"].to(device, non_blocking=True) if use_seg_ids else None
        sam_features = batch["sam_features"].to(device, non_blocking=True) if use_sam_features else None

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            img_emb, txt_emb = model(
                images,
                input_ids,
                attention_mask,
                seg_ids=seg_ids,
                sam_features=sam_features,
            )

        img_embeds_list.append(img_emb.float().cpu())
        txt_embeds_list.append(txt_emb.float().cpu())
        image_ids_list.append(batch["image_id"].cpu())
        sentids_list.append(batch["sentid"].cpu())

        if idx % 20 == 0:
            logger.info("Encoded %d/%d batches", idx, len(loader))

    img_embeds = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    image_ids = torch.cat(image_ids_list, dim=0)
    sentids = torch.cat(sentids_list, dim=0)

    seen_image_ids = set()
    first_occurrence_indices = []
    unique_image_ids_list = []
    for idx in range(len(image_ids)):
        img_id = image_ids[idx].item()
        if img_id not in seen_image_ids:
            seen_image_ids.add(img_id)
            first_occurrence_indices.append(idx)
            unique_image_ids_list.append(img_id)

    unique_image_ids = torch.tensor(unique_image_ids_list, dtype=image_ids.dtype)
    img_embeds_unique = img_embeds[first_occurrence_indices]
    return img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, unique_image_ids_list


def compute_standard_metrics(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims: torch.Tensor) -> dict:
    r_t2i, r_i2t = compute_recall_at_k(
        img_embeds_unique,
        txt_embeds,
        image_ids,
        unique_image_ids,
        sims=sims,
    )

    sims_np = sims.t().numpy()
    unique_image_ids_list = unique_image_ids.tolist()
    caption_ids = list(range(txt_embeds.shape[0]))
    i2t_ranked, t2i_ranked = build_ranked_dicts(sims_np, unique_image_ids_list, caption_ids)

    _, caption_to_image_idx, image_to_caption_indices = _build_gt_mappings(image_ids, unique_image_ids)
    gt_i2t = {
        unique_image_ids_list[img_idx]: set(cap_indices)
        for img_idx, cap_indices in image_to_caption_indices.items()
    }
    gt_t2i = {
        cap_idx: {unique_image_ids_list[caption_to_image_idx[cap_idx].item()]}
        for cap_idx in range(len(caption_to_image_idx))
    }
    mapr_rprec = compute_mapr_rprecision(i2t_ranked, t2i_ranked, gt_i2t, gt_t2i)

    return {
        "r1_i2t": r_i2t[1],
        "r5_i2t": r_i2t[5],
        "r10_i2t": r_i2t[10],
        "r1_t2i": r_t2i[1],
        "r5_t2i": r_t2i[5],
        "r10_t2i": r_t2i[10],
        "mapr_i2t": mapr_rprec["mapr_i2t"],
        "mapr_t2i": mapr_rprec["mapr_t2i"],
        "rprecision_i2t": mapr_rprec["rprecision_i2t"],
        "rprecision_t2i": mapr_rprec["rprecision_t2i"],
    }


def compute_metrics(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, unique_image_ids_list, eval_dataset: str) -> dict:
    logger.info("Computing similarity matrix: %d images x %d captions", img_embeds_unique.shape[0], txt_embeds.shape[0])
    sims = chunked_matmul(img_embeds_unique, txt_embeds)
    metrics = compute_standard_metrics(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims)

    if eval_dataset == "coco":
        sims_np = sims.t().numpy()
        eccv_scores = compute_eccv_metrics(
            sims_np,
            image_ids=unique_image_ids_list,
            caption_ids=sentids.tolist(),
            dataset="coco",
        )
        if eccv_scores:
            metrics.update(
                {
                    "coco_5k_r1_i2t": eccv_scores.get("coco_5k_r1", {}).get("i2t", 0),
                    "coco_5k_r1_t2i": eccv_scores.get("coco_5k_r1", {}).get("t2i", 0),
                    "coco_5k_r5_i2t": eccv_scores.get("coco_5k_r5", {}).get("i2t", 0),
                    "coco_5k_r5_t2i": eccv_scores.get("coco_5k_r5", {}).get("t2i", 0),
                    "coco_5k_r10_i2t": eccv_scores.get("coco_5k_r10", {}).get("i2t", 0),
                    "coco_5k_r10_t2i": eccv_scores.get("coco_5k_r10", {}).get("t2i", 0),
                    "eccv_map_at_r_i2t": eccv_scores.get("eccv_map_at_r", {}).get("i2t", 0),
                    "eccv_map_at_r_t2i": eccv_scores.get("eccv_map_at_r", {}).get("t2i", 0),
                    "eccv_rprecision_i2t": eccv_scores.get("eccv_rprecision", {}).get("i2t", 0),
                    "eccv_rprecision_t2i": eccv_scores.get("eccv_rprecision", {}).get("t2i", 0),
                    "cxc_r1_i2t": eccv_scores.get("cxc_r1", {}).get("i2t", 0),
                    "cxc_r1_t2i": eccv_scores.get("cxc_r1", {}).get("t2i", 0),
                    "cxc_r5_i2t": eccv_scores.get("cxc_r5", {}).get("i2t", 0),
                    "cxc_r5_t2i": eccv_scores.get("cxc_r5", {}).get("t2i", 0),
                    "cxc_r10_i2t": eccv_scores.get("cxc_r10", {}).get("i2t", 0),
                    "cxc_r10_t2i": eccv_scores.get("cxc_r10", {}).get("t2i", 0),
                }
            )
        else:
            logger.warning("ECCV/CxC metrics unavailable; standard COCO R@K metrics are still present.")
    return metrics


def output_json_path(cache_dir: Path, run_id: str, train_dataset: str, eval_dataset: str, seed: int) -> Path:
    safe_run = run_id.replace("/", "_")
    return cache_dir / f"{safe_run}_{train_dataset}_to_{eval_dataset}_s{seed}.json"


def log_to_wandb(results: dict, wandb_run_id: str, project: str) -> None:
    import wandb

    wandb.init(id=wandb_run_id, project=project, resume="must")
    eval_dataset = results["eval_dataset"]
    train_dataset = results["train_dataset"]
    for key, value in results["metrics"].items():
        wandb.run.summary[f"ood/{train_dataset}_to_{eval_dataset}/{key}"] = value
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset OOD retrieval evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, default=None, choices=["coco", "flickr30k", "flickr"])
    parser.add_argument("--eval-dataset", type=str, default=None, choices=["coco", "flickr30k", "flickr"])
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="/Volumes/T7/Research/experiments/datasets")
    parser.add_argument("--cache-dir", type=str, default="results/cache/ood_eval")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
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
    identity = parse_identity(args.checkpoint, checkpoint_config, args.wandb_run_name)
    if args.train_dataset is not None:
        identity["train_dataset"] = DATASET_ALIASES[args.train_dataset]
    if identity["train_dataset"] not in OPPOSITE_DATASET:
        raise ValueError(f"Could not determine train dataset from checkpoint/config: {identity}")
    eval_dataset = DATASET_ALIASES[args.eval_dataset] if args.eval_dataset else OPPOSITE_DATASET[identity["train_dataset"]]
    if eval_dataset == identity["train_dataset"]:
        raise ValueError("eval_dataset must be different from train_dataset for OOD evaluation.")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_json_path(
        cache_dir,
        identity["run_id"],
        identity["train_dataset"],
        eval_dataset,
        identity["seed"],
    )
    if output_path.exists() and not args.force:
        logger.info("Cache exists, skipping: %s", output_path)
        return

    eval_config = build_eval_config(checkpoint_config, eval_dataset, args.data_root, args.batch_size, args.num_workers)
    tokenizer = build_tokenizer(checkpoint_config)
    loader = create_image_text_dataloader(eval_config, tokenizer, split="test")
    logger.info(
        "Evaluating %s trained on %s (seed=%s) on %s test: %d caption samples",
        identity["run_id"],
        identity["train_dataset"],
        identity["seed"],
        eval_dataset,
        len(loader.dataset),
    )

    use_amp = device.type == "cuda"
    embeddings = extract_embeddings(model, loader, device, use_amp=use_amp)
    metrics = compute_metrics(*embeddings, eval_dataset=eval_dataset)

    results = {
        **identity,
        "eval_dataset": eval_dataset,
        "checkpoint_path": args.checkpoint,
        "n_images": int(embeddings[3].shape[0]),
        "n_captions": int(embeddings[1].shape[0]),
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)
    for key, value in metrics.items():
        logger.info("  %s: %.4f", key, value)

    if args.log_wandb:
        if not args.wandb_run_id:
            raise ValueError("--log-wandb requires --wandb_run_id")
        project = args.wandb_project or checkpoint_config.get("logging", {}).get("wandb_project", "clip-retrieval")
        log_to_wandb(results, args.wandb_run_id, project)
        logger.info("Logged OOD metrics to WandB summary with ood/* prefix.")


if __name__ == "__main__":
    main()
