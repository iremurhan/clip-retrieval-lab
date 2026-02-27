"""
tools/eval.py
-------------
Unified evaluation script for DualEncoder models.

Two modes:
  1. Standard mode (default): Evaluates on COCO/Flickr30k using R@K + MAP@K
     via src.metrics.
  2. ECCV mode (--eccv flag): Runs the ECCV Caption protocol (naver-ai/eccv-caption)
     on the COCO Karpathy test split (5000 images, 25000 captions).

Usage:
    # Standard evaluation on test set
    python tools/eval.py \\
        --config configs/config_coco.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --split test

    # ECCV Caption benchmark
    python tools/eval.py \\
        --config configs/config_eccv.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --eccv

    # Zero-shot (no checkpoint)
    python tools/eval.py \\
        --config configs/config_coco.yaml \\
        --split test
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.setup import setup_config, setup_seed
from src.model import DualEncoder
from src.data import build_eval_transform, create_image_text_dataloader
from src.metrics import compute_recall_at_k, compute_map_at_k

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(config, checkpoint_path, device):
    """
    Load DualEncoder, optionally restoring weights from a checkpoint.

    Always uses strict=False so that both fine-tuned and custom (FNE)
    checkpoints can be loaded without key-mismatch errors.  Missing and
    unexpected keys are logged as warnings.

    Args:
        config: Merged config dict.
        checkpoint_path: Path to .pth file, or None for zero-shot.
        device: torch.device.

    Returns:
        model in eval mode on *device*.
    """
    model = DualEncoder(config).to(device)

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint (strict=False): {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(
                f"Missing keys ({len(result.missing_keys)}): "
                f"{result.missing_keys[:10]}{'...' if len(result.missing_keys) > 10 else ''}"
            )
        if result.unexpected_keys:
            logger.warning(
                f"Unexpected keys ({len(result.unexpected_keys)}): "
                f"{result.unexpected_keys[:10]}{'...' if len(result.unexpected_keys) > 10 else ''}"
            )
    else:
        logger.info("No checkpoint provided — using zero-shot CLIP weights.")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Standard Evaluation  (R@K, MAP@K via src.metrics)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_standard_eval(model, config, tokenizer, split, device):
    """
    Standard evaluation replicating Trainer.evaluate logic.

    Loads data via create_image_text_dataloader, extracts embeddings with
    AMP, deduplicates images by image_id, and computes Recall@K + MAP@K.

    Returns:
        metrics: dict with all computed metric values.
    """
    logger.info(f"Standard evaluation on split='{split}'...")

    loader = create_image_text_dataloader(config, tokenizer, split=split)
    logger.info(f"DataLoader created: {len(loader)} batches, {len(loader.dataset)} samples.")

    img_embeds_list = []
    txt_embeds_list = []
    image_ids_list = []

    use_amp = torch.cuda.is_available()

    for batch in tqdm(loader, desc=f"Eval [{split}]"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            img_emb, txt_emb = model(images, input_ids, attention_mask)

        img_embeds_list.append(img_emb.cpu())
        txt_embeds_list.append(txt_emb.cpu())
        image_ids_list.append(batch["image_id"].cpu())

    img_embeds = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    image_ids = torch.cat(image_ids_list, dim=0)

    # --- Deduplicate images (keep first occurrence per image_id) ---
    seen = set()
    first_indices = []
    unique_ids_list = []
    for idx in range(len(image_ids)):
        iid = image_ids[idx].item()
        if iid not in seen:
            seen.add(iid)
            first_indices.append(idx)
            unique_ids_list.append(iid)

    unique_image_ids = torch.tensor(unique_ids_list, dtype=image_ids.dtype)
    img_embeds_unique = img_embeds[first_indices]

    logger.info(
        f"Embeddings: {img_embeds_unique.shape[0]} unique images, "
        f"{txt_embeds.shape[0]} captions, dim={txt_embeds.shape[1]}"
    )

    # --- Metrics ---
    r_t2i, r_i2t = compute_recall_at_k(
        img_embeds_unique, txt_embeds, image_ids, unique_image_ids
    )
    map_t2i, map_i2t = compute_map_at_k(
        img_embeds_unique, txt_embeds, image_ids, unique_image_ids, k_values=[5, 10]
    )

    metrics = {
        "t2i_r1": r_t2i[1], "t2i_r5": r_t2i[5], "t2i_r10": r_t2i[10],
        "i2t_r1": r_i2t[1], "i2t_r5": r_i2t[5], "i2t_r10": r_i2t[10],
        "t2i_map5": map_t2i[5], "t2i_map10": map_t2i[10],
        "i2t_map5": map_i2t[5], "i2t_map10": map_i2t[10],
    }
    return metrics


def print_standard_results(metrics, split):
    """Pretty-print standard Recall@K and MAP@K results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  Standard Evaluation Results — split={split}")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<20s} {'I2T':>10s}   {'T2I':>10s}")
    logger.info(f"  {'-'*20} {'-'*10}   {'-'*10}")
    logger.info(f"  {'R@1':<20s} {metrics['i2t_r1']:>10.2f}   {metrics['t2i_r1']:>10.2f}")
    logger.info(f"  {'R@5':<20s} {metrics['i2t_r5']:>10.2f}   {metrics['t2i_r5']:>10.2f}")
    logger.info(f"  {'R@10':<20s} {metrics['i2t_r10']:>10.2f}   {metrics['t2i_r10']:>10.2f}")
    logger.info(f"  {'MAP@5':<20s} {metrics['i2t_map5']:>10.2f}   {metrics['t2i_map5']:>10.2f}")
    logger.info(f"  {'MAP@10':<20s} {metrics['i2t_map10']:>10.2f}   {metrics['t2i_map10']:>10.2f}")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# ECCV Caption Evaluation  (via eccv_caption package)
# ═══════════════════════════════════════════════════════════════════════════

def load_eccv_test_data(karpathy_json_path, images_root, eccv_coco_ids):
    """
    Load ECCV test subset from Karpathy JSON, ordered by eccv_coco_ids.

    eccv_caption.Metrics().coco_ids provides the canonical caption IDs
    (COCO annotation sentids).  This function maps them to Karpathy JSON
    entries and returns ordered images + captions.

    Returns:
        ordered_images: list of (image_id, image_path) — unique, encounter order
        ordered_captions: list of (caption_id, caption_text) — eccv_coco_ids order
    """
    logger.info(f"Loading Karpathy JSON: {karpathy_json_path}")
    with open(karpathy_json_path, "r") as f:
        data = json.load(f)

    # Build sentid → entry lookup (test split only)
    sentid_lookup = {}
    for img_entry in data["images"]:
        if img_entry["split"] != "test":
            continue

        img_id = int(
            img_entry.get("cocoid")
            or img_entry.get("imgid")
            or img_entry.get("id")
        )

        filepath = img_entry.get("filepath", "")
        filename = img_entry.get("filename", "")

        for sent in img_entry["sentences"]:
            sentid_lookup[int(sent["sentid"])] = {
                "image_id": img_id,
                "caption": sent["raw"],
                "filepath": filepath,
                "filename": filename,
            }

    eccv_ids = eccv_coco_ids.tolist() if isinstance(eccv_coco_ids, np.ndarray) else list(eccv_coco_ids)

    ordered_captions = []
    seen_iids = set()
    ordered_images = []
    missing = 0

    for cid in eccv_ids:
        entry = sentid_lookup.get(cid)
        if entry is None:
            missing += 1
            continue

        ordered_captions.append((cid, entry["caption"]))

        iid = entry["image_id"]
        if iid not in seen_iids:
            seen_iids.add(iid)
            fp = entry["filepath"].strip()
            fn = entry["filename"]
            img_path = os.path.join(images_root, fp, fn) if fp else os.path.join(images_root, fn)
            if not os.path.exists(img_path):
                img_path = os.path.join(images_root, fn)
            ordered_images.append((iid, img_path))

    logger.info(
        f"ECCV data: {len(ordered_images)} images, "
        f"{len(ordered_captions)} captions (missing: {missing})"
    )
    return ordered_images, ordered_captions


@torch.no_grad()
def run_eccv_eval(model, config, tokenizer, device):
    """
    ECCV Caption protocol evaluation.

    1. Load canonical caption IDs from eccv_caption.
    2. Filter Karpathy JSON to 5000 images / 25000 captions in exact order.
    3. Compute L2-normalised embeddings with AMP.
    4. Build 5000 × 25000 similarity matrix (single GPU matmul).
    5. Construct i2t / t2i top-K dicts.
    6. Delegate to eccv_caption.Metrics.compute_all_metrics().

    Returns:
        scores: dict from eccv_caption (eccv_r1, eccv_map_at_r, coco_*, cxc_*, etc.)
    """
    from eccv_caption import Metrics as ECCVMetrics

    logger.info("ECCV Caption protocol evaluation...")
    eccv_metric = ECCVMetrics()
    eccv_coco_ids = eccv_metric.coco_ids
    logger.info(f"ECCV canonical caption IDs: {len(eccv_coco_ids)}")

    images_root = config["data"]["images_path"]
    captions_path = config["data"]["captions_path"]
    ordered_images, ordered_captions = load_eccv_test_data(
        captions_path, images_root, eccv_coco_ids
    )

    image_size = config["data"]["image_size"]
    transform = build_eval_transform(image_size)
    max_length = config["data"]["max_length"]
    use_amp = torch.cuda.is_available()
    batch_size = config.get("training", {}).get("batch_size", 64)

    # --- Image embeddings ---
    logger.info(f"Computing image embeddings ({len(ordered_images)} images)...")
    img_embeds_list = []
    all_iids = []

    for i in tqdm(range(0, len(ordered_images), batch_size), desc="Images"):
        batch_imgs = []
        for j in range(i, min(i + batch_size, len(ordered_images))):
            iid, img_path = ordered_images[j]
            image = Image.open(img_path).convert("RGB")
            batch_imgs.append(transform(image))
            all_iids.append(iid)

        pixel_values = torch.stack(batch_imgs, dim=0).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            img_emb = model.encode_image(pixel_values)
        img_embeds_list.append(img_emb.cpu())

    img_embeds = torch.cat(img_embeds_list, dim=0)

    # --- Text embeddings ---
    logger.info(f"Computing text embeddings ({len(ordered_captions)} captions)...")
    txt_embeds_list = []
    all_cids = []

    for i in tqdm(range(0, len(ordered_captions), batch_size), desc="Captions"):
        batch_texts = []
        for j in range(i, min(i + batch_size, len(ordered_captions))):
            cid, caption = ordered_captions[j]
            all_cids.append(cid)
            batch_texts.append(caption)

        tokenized = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            txt_emb = model.encode_text(input_ids, attention_mask)
        txt_embeds_list.append(txt_emb.cpu())

    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    logger.info(f"Embeddings: img={img_embeds.shape}, txt={txt_embeds.shape}")

    # --- Similarity matrix (single matmul on GPU) ---
    logger.info("Computing 5000 × 25000 similarity matrix...")
    img_gpu = img_embeds.to(device)
    txt_gpu = txt_embeds.to(device)
    sims = img_gpu @ txt_gpu.T  # [N_images, N_captions]
    logger.info(f"Sim matrix: {sims.shape}")

    # --- Build i2t / t2i dicts (top-K = 50) ---
    K = 50
    all_iids_np = np.array(all_iids)
    all_cids_np = np.array(all_cids)

    i2t = {}
    for idx, iid in enumerate(all_iids_np):
        _, indices = sims[idx, :].topk(K)
        i2t[int(iid)] = [int(c) for c in all_cids_np[indices.cpu().numpy()]]

    t2i = {}
    for idx, cid in enumerate(all_cids_np):
        _, indices = sims[:, idx].topk(K)
        t2i[int(cid)] = [int(i) for i in all_iids_np[indices.cpu().numpy()]]

    # Free GPU
    del img_gpu, txt_gpu, sims
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Delegate to eccv_caption ---
    logger.info("Running eccv_caption.compute_all_metrics()...")
    scores = eccv_metric.compute_all_metrics(
        i2t, t2i,
        target_metrics=(
            "eccv_r1", "eccv_map_at_r", "eccv_rprecision",
            "coco_1k_recalls", "coco_5k_recalls", "cxc_recalls",
        ),
        Ks=(1, 5, 10),
        verbose=False,
    )
    return scores


def print_eccv_results(scores):
    """Pretty-print ECCV Caption benchmark results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  ECCV Caption Benchmark Results")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<25s} {'I2T':>10s}   {'T2I':>10s}")
    logger.info(f"  {'-'*25} {'-'*10}   {'-'*10}")

    for metric_name in sorted(scores.keys()):
        val = scores[metric_name]
        if isinstance(val, dict):
            i2t = val.get("i2t", "—")
            t2i = val.get("t2i", "—")
            i2t_s = f"{i2t:.4f}" if isinstance(i2t, float) else str(i2t)
            t2i_s = f"{t2i:.4f}" if isinstance(t2i, float) else str(t2i)
            logger.info(f"  {metric_name:<25s} {i2t_s:>10s}   {t2i_s:>10s}")

    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# WandB Logging Hook
# ═══════════════════════════════════════════════════════════════════════════

def log_results_to_wandb(results, mode, config, run_name):
    """
    Optional WandB logging hook.

    For standard mode, logs flat metric dict.
    For ECCV mode, logs flat metrics + a WandB Table for visual comparison.

    Args:
        results: dict of metric results.
        mode: "standard" or "eccv".
        config: merged config dict.
        run_name: WandB run name.
    """
    import wandb

    project = config.get("logging", {}).get("wandb_project", "retrieval-benchmark")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    wandb.init(
        project=project,
        name=run_name or f"{job_id}_{mode}_eval",
        config={"mode": mode, "evaluation": True},
        job_type="benchmark",
    )

    if mode == "standard":
        wandb.log(results)
        for k, v in results.items():
            wandb.run.summary[k] = v
    else:
        # ECCV: flatten nested dicts
        flat = {}
        for metric_name, directions in results.items():
            if isinstance(directions, dict):
                for d, v in directions.items():
                    flat[f"{metric_name}/{d}"] = v

        wandb.log(flat)

        # WandB Table
        table = wandb.Table(columns=["metric", "i2t", "t2i"])
        for metric_name, directions in sorted(results.items()):
            if isinstance(directions, dict):
                i2t = round(directions.get("i2t", 0), 4)
                t2i = round(directions.get("t2i", 0), 4)
                table.add_data(metric_name, i2t, t2i)
        wandb.log({"eccv_results": table})

        for key in ("eccv_r1", "eccv_map_at_r", "coco_5k_r1"):
            if key in results and isinstance(results[key], dict):
                for d, v in results[key].items():
                    wandb.run.summary[f"{key}_{d}"] = v

    wandb.finish()
    logger.info(f"Results logged to WandB project '{project}'.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified DualEncoder evaluation script (Standard + ECCV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config path (e.g. configs/config_coco.yaml or configs/config_eccv.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (.pth). Omit for zero-shot.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Data split for standard evaluation (default: test)")
    parser.add_argument("--eccv", action="store_true",
                        help="Enable ECCV Caption protocol evaluation")
    parser.add_argument("--run_name", type=str, default=None,
                        help="WandB run name (optional)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides as key=value")
    args = parser.parse_args()

    # --- Config ---
    config = setup_config(config_path=args.config, overrides=args.override)
    config.setdefault("data", {})
    config["data"].setdefault("max_length", 77)
    config["data"].setdefault("num_workers", 8)
    config["data"].setdefault("batch_size", 256)
    config["data"].setdefault("image_size", 336)
    setup_seed(config.get("training", {}).get("seed", 42))

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Model ---
    model = load_model(config, args.checkpoint, device)

    # --- Tokenizer ---
    model_name = config["model"]["image_model_name"]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # --- Evaluate ---
    use_wandb = config.get("logging", {}).get("use_wandb", True)
    debug_mode = config.get("debug", {}).get("debug_mode", False)

    if args.eccv:
        # ── ECCV Caption Protocol ──
        scores = run_eccv_eval(model, config, tokenizer, device)
        print_eccv_results(scores)

        if use_wandb and not debug_mode:
            log_results_to_wandb(scores, "eccv", config, args.run_name)
    else:
        # ── Standard R@K + MAP@K ──
        metrics = run_standard_eval(model, config, tokenizer, args.split, device)
        print_standard_results(metrics, args.split)

        if use_wandb and not debug_mode:
            log_results_to_wandb(metrics, "standard", config, args.run_name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
