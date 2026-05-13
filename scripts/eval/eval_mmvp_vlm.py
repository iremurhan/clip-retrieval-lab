from __future__ import annotations

import argparse
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
from src.data import build_eval_transform
from src.eval.mmvp_vlm import evaluate_mmvp_vlm


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


def output_json_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    return cache_dir / f"{run_id.replace('/', '_')}_{dataset}_s{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on MMVP-VLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/Volumes/T7/Research/experiments/datasets/mmvp_vlm")
    parser.add_argument("--cache-dir", type=str, default="results/cache/mmvp_vlm")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
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
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    identity = parse_identity(args.checkpoint, config, args.wandb_run_name)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else output_json_path(cache_dir, identity["run_id"], identity["dataset"], identity["seed"])
    if output.exists() and not args.force:
        logger.info("Cache exists, skipping: %s", output)
        return

    tokenizer = build_tokenizer(config)
    transform = build_eval_transform(config["data"]["image_size"])
    summary, pair_results = evaluate_mmvp_vlm(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device=device,
        data_dir=args.data_dir,
        max_length=config["data"].get("max_length", 77),
    )

    payload = {
        **identity,
        "checkpoint_path": args.checkpoint,
        "metrics": summary,
        "pairs": pair_results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output)

    if args.log_wandb:
        if not args.wandb_run_id:
            raise ValueError("--log-wandb requires --wandb_run_id")
        import wandb

        project = args.wandb_project or config.get("logging", {}).get("wandb_project", "clip-retrieval")
        wandb.init(id=args.wandb_run_id, project=project, resume="must")
        for key, value in summary.items():
            wandb.run.summary[f"mmvp_vlm/{key}"] = value
        wandb.finish()
        logger.info("Logged MMVP-VLM metrics to WandB summary.")


if __name__ == "__main__":
    main()
