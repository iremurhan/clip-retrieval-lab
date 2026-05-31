from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}


def build_tokenizer(config: dict):
    if config.get("model", {}).get("text_encoder") == "blip":
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(config["model"]["text_model_name"])
    from transformers import CLIPTokenizer

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
    dataset: str,
    data_root: str | Path | None,
    batch_size: int | None,
    num_workers: int | None,
) -> dict:
    config = copy.deepcopy(checkpoint_config)
    if data_root is not None:
        config.setdefault("data", {}).update(dataset_paths(dataset, data_root))
    else:
        config.setdefault("data", {})["dataset"] = dataset
    config.setdefault("debug", {})["debug_mode"] = False
    if batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(batch_size)
    if num_workers is not None:
        config.setdefault("data", {})["num_workers"] = int(num_workers)
    return config


def compute_standard_metrics(
    img_embeds_unique,
    txt_embeds,
    image_ids,
    unique_image_ids,
    sims,
    prefix: str = "test",
) -> dict[str, float]:
    from src.metrics import _build_gt_mappings, build_ranked_dicts, compute_mapr_rprecision, compute_recall_at_k

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
        f"{prefix}/r1_i2t": r_i2t[1],
        f"{prefix}/r5_i2t": r_i2t[5],
        f"{prefix}/r10_i2t": r_i2t[10],
        f"{prefix}/r1_t2i": r_t2i[1],
        f"{prefix}/r5_t2i": r_t2i[5],
        f"{prefix}/r10_t2i": r_t2i[10],
        f"{prefix}/mapr_i2t": mapr_rprec["mapr_i2t"],
        f"{prefix}/mapr_t2i": mapr_rprec["mapr_t2i"],
        f"{prefix}/rprecision_i2t": mapr_rprec["rprecision_i2t"],
        f"{prefix}/rprecision_t2i": mapr_rprec["rprecision_t2i"],
    }


def compute_test_metrics(embeddings: tuple, dataset: str) -> dict[str, float]:
    from src.metrics import compute_eccv_metrics
    from src.utils import chunked_matmul

    img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, unique_image_ids_list = embeddings
    logger.info(
        "Computing similarity matrix: %d images x %d captions",
        img_embeds_unique.shape[0],
        txt_embeds.shape[0],
    )
    sims = chunked_matmul(img_embeds_unique, txt_embeds)
    metrics = compute_standard_metrics(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims, prefix="test")

    if dataset == "coco":
        eccv_scores = compute_eccv_metrics(
            sims.t().numpy(),
            image_ids=unique_image_ids_list,
            caption_ids=sentids.tolist(),
            dataset="coco",
        )
        if eccv_scores:
            metrics.update(
                {
                    "test/coco_5k_r1_i2t": eccv_scores.get("coco_5k_r1", {}).get("i2t", 0),
                    "test/coco_5k_r1_t2i": eccv_scores.get("coco_5k_r1", {}).get("t2i", 0),
                    "test/coco_5k_r5_i2t": eccv_scores.get("coco_5k_r5", {}).get("i2t", 0),
                    "test/coco_5k_r5_t2i": eccv_scores.get("coco_5k_r5", {}).get("t2i", 0),
                    "test/coco_5k_r10_i2t": eccv_scores.get("coco_5k_r10", {}).get("i2t", 0),
                    "test/coco_5k_r10_t2i": eccv_scores.get("coco_5k_r10", {}).get("t2i", 0),
                    "test/coco_1k_r1_i2t": eccv_scores.get("coco_1k_r1", {}).get("i2t", 0),
                    "test/coco_1k_r1_t2i": eccv_scores.get("coco_1k_r1", {}).get("t2i", 0),
                    "test/coco_1k_r5_i2t": eccv_scores.get("coco_1k_r5", {}).get("i2t", 0),
                    "test/coco_1k_r5_t2i": eccv_scores.get("coco_1k_r5", {}).get("t2i", 0),
                    "test/coco_1k_r10_i2t": eccv_scores.get("coco_1k_r10", {}).get("i2t", 0),
                    "test/coco_1k_r10_t2i": eccv_scores.get("coco_1k_r10", {}).get("t2i", 0),
                    "test/eccv_map_at_r_i2t": eccv_scores.get("eccv_map_at_r", {}).get("i2t", 0),
                    "test/eccv_map_at_r_t2i": eccv_scores.get("eccv_map_at_r", {}).get("t2i", 0),
                    "test/eccv_rprecision_i2t": eccv_scores.get("eccv_rprecision", {}).get("i2t", 0),
                    "test/eccv_rprecision_t2i": eccv_scores.get("eccv_rprecision", {}).get("t2i", 0),
                    "test/cxc_r1_i2t": eccv_scores.get("cxc_r1", {}).get("i2t", 0),
                    "test/cxc_r1_t2i": eccv_scores.get("cxc_r1", {}).get("t2i", 0),
                    "test/cxc_r5_i2t": eccv_scores.get("cxc_r5", {}).get("i2t", 0),
                    "test/cxc_r5_t2i": eccv_scores.get("cxc_r5", {}).get("t2i", 0),
                    "test/cxc_r10_i2t": eccv_scores.get("cxc_r10", {}).get("i2t", 0),
                    "test/cxc_r10_t2i": eccv_scores.get("cxc_r10", {}).get("t2i", 0),
                }
            )
        else:
            logger.warning("eccv_caption unavailable or returned no scores; standard COCO test metrics are still present.")
    return metrics


def output_json_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    return cache_dir / f"{run_id.replace('/', '_')}_{dataset}_s{seed}.json"


def log_to_wandb(metrics: dict[str, float], wandb_run_id: str, project: str) -> None:
    import wandb

    wandb.init(id=wandb_run_id, project=project, resume="must")
    for key, value in metrics.items():
        wandb.run.summary[key] = value
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone in-domain retrieval evaluation for trained checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, choices=["coco", "flickr30k", "flickr"])
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--cache-dir", type=str, default=str(ARTIFACT_ROOT / "cache" / "in_domain_retrieval"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print metrics, but do not write to W&B.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from scripts.eval.eval_cross_dataset import extract_embeddings, select_device
    from scripts.eval.eval_sugarcrepe import load_model_from_checkpoint
    from src.data import create_image_text_dataloader

    device = select_device(args.device)
    logger.info("Device: %s", device)
    model, checkpoint_config = load_model_from_checkpoint(args.checkpoint, device)
    identity = parse_identity(args.checkpoint, checkpoint_config, args.wandb_run_name)
    dataset = DATASET_ALIASES[args.dataset] if args.dataset else identity["dataset"]
    if dataset not in {"coco", "flickr30k"}:
        raise ValueError(f"Could not determine evaluation dataset: {dataset!r}")

    eval_config = build_eval_config(
        checkpoint_config,
        dataset=dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    tokenizer = build_tokenizer(checkpoint_config)
    loader = create_image_text_dataloader(eval_config, tokenizer, split="test")
    logger.info(
        "Evaluating %s dataset=%s seed=%s on %d caption samples",
        identity["run_id"],
        dataset,
        identity["seed"],
        len(loader.dataset),
    )

    embeddings = extract_embeddings(model, loader, device, use_amp=(device.type == "cuda"))
    metrics = compute_test_metrics(embeddings, dataset=dataset)

    payload = {
        **identity,
        "dataset": dataset,
        "checkpoint_path": args.checkpoint,
        "n_images": int(embeddings[3].shape[0]),
        "n_captions": int(embeddings[1].shape[0]),
        "metrics": metrics,
    }

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else output_json_path(
        cache_dir,
        identity["run_id"],
        dataset,
        identity["seed"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output)

    print("\nMetrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")

    if args.log_wandb:
        if not args.wandb_run_id:
            raise ValueError("--log-wandb requires --wandb_run_id")
        project = args.wandb_project or checkpoint_config.get("logging", {}).get("wandb_project", "clip-retrieval")
        if args.dry_run:
            print(f"\n[DRY RUN] Would log {len(metrics)} metrics to W&B run {args.wandb_run_id} in project {project}:")
            for key in sorted(metrics):
                print(f"  {key}={metrics[key]:.4f}")
        else:
            log_to_wandb(metrics, args.wandb_run_id, project)
            logger.info("Logged in-domain retrieval metrics to WandB summary.")


if __name__ == "__main__":
    main()
