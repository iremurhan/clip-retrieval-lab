"""
tools/eval.py
-------------
ECCV Caption benchmark evaluation for DualEncoder models.

Computes all retrieval metrics (ECCV, COCO 1K/5K, CxC) via the eccv_caption
package on the COCO Karpathy test split (5000 images, 25000 captions).

Checkpoint handling:
    .pth file   → load_state_dict(strict=False) onto DualEncoder
    HF model ID → zero-shot, no weight loading (DualEncoder uses pretrained)

Usage:
    # Fine-tuned checkpoint
    python tools/eval.py --config configs/config_coco.yaml \\
        --checkpoint /path/to/best_model.pth

    # Zero-shot (HuggingFace model)
    python tools/eval.py --config configs/config_coco.yaml \\
        --checkpoint openai/clip-vit-large-patch14-336
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
from eccv_caption import Metrics as ECCVMetrics

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.setup import setup_config, setup_seed
from src.model import DualEncoder
from src.data import build_eval_transform

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

def load_model(config, checkpoint, device):
    """
    Build DualEncoder and optionally load fine-tuned weights.

    If *checkpoint* ends with `.pth`, weights are loaded via
    load_state_dict(strict=False).  Otherwise *checkpoint* is treated as a
    HuggingFace model identifier (e.g. ``openai/clip-vit-large-patch14-336``)
    and the config's ``image_model_name`` is overridden so that DualEncoder
    loads pretrained weights directly — no load_state_dict call is made.

    Args:
        config: Merged config dict (mutated in-place for zero-shot override).
        checkpoint: .pth path **or** HuggingFace model ID **or** None.
        device: torch.device.

    Returns:
        DualEncoder in eval mode on *device*.
    """
    is_checkpoint = checkpoint is not None and checkpoint.endswith(".pth")
    is_zeroshot_override = checkpoint is not None and not checkpoint.endswith(".pth")

    if is_zeroshot_override:
        logger.info(f"Zero-shot override: setting model to '{checkpoint}'")
        config["model"]["image_model_name"] = checkpoint

    model = DualEncoder(config).to(device)

    if is_checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        logger.info(f"Loading checkpoint (strict=False): {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")
    else:
        logger.info("Using pretrained weights (zero-shot).")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading — COCO Karpathy test filtered by eccv_caption IDs
# ═══════════════════════════════════════════════════════════════════════════

def load_eccv_test_data(karpathy_json_path, images_root, eccv_coco_ids):
    """
    Extract 5000 images and 25000 captions from Karpathy JSON, ordered by
    the canonical caption IDs from ``eccv_caption.Metrics().coco_ids``.

    Args:
        karpathy_json_path: Path to ``dataset_coco.json`` (Karpathy format).
        images_root: Root directory for COCO images.
        eccv_coco_ids: Ordered caption IDs from eccv_caption.

    Returns:
        ordered_images:   list[(image_id, image_path)]  — unique, encounter order
        ordered_captions: list[(caption_id, caption_text)] — eccv_coco_ids order
    """
    logger.info(f"Loading Karpathy JSON: {karpathy_json_path}")
    with open(karpathy_json_path, "r") as f:
        data = json.load(f)

    # Build sentid → entry lookup (test split only)
    sentid_lookup = {}
    for img_entry in data["images"]:
        if img_entry["split"] != "test":
            continue

        _cocoid = img_entry.get("cocoid")
        _imgid = img_entry.get("imgid")
        _id = img_entry.get("id")
        if _cocoid is not None:
            img_id = int(_cocoid)
        elif _imgid is not None:
            img_id = int(_imgid)
        elif _id is not None:
            img_id = int(_id)
        else:
            logger.warning(f"Skipping image entry with no ID: {img_entry.get('filename')}")
            continue
        filepath = img_entry.get("filepath", "")
        filename = img_entry.get("filename", "")

        for sent in img_entry["sentences"]:
            sentid_lookup[int(sent["sentid"])] = {
                "image_id": img_id,
                "caption": sent["raw"],
                "filepath": filepath,
                "filename": filename,
            }

    # Iterate in eccv canonical order
    eccv_ids = eccv_coco_ids.tolist() if isinstance(eccv_coco_ids, np.ndarray) else list(eccv_coco_ids)

    ordered_captions = []
    seen_iids = set()
    ordered_images = []
    missing = 0

    for cid in eccv_ids:
        entry = sentid_lookup.get(cid)
        if entry is None:
            missing += 1
            if missing <= 5:
                logger.error(f"ECCV caption ID {cid} not found in Karpathy test split.")
            continue

        ordered_captions.append((cid, entry["caption"]))

        iid = entry["image_id"]
        if iid not in seen_iids:
            seen_iids.add(iid)
            fp = entry["filepath"].strip()
            fn = entry["filename"]
            img_path = os.path.join(images_root, fp, fn) if fp else os.path.join(images_root, fn)
            if not os.path.exists(img_path):
                fallback = os.path.join(images_root, fn)
                logger.warning(f"Path not found: {img_path} → falling back to {fallback}")
                img_path = fallback
            ordered_images.append((iid, img_path))

    logger.info(f"ECCV data: {len(ordered_images)} images, {len(ordered_captions)} captions (missing: {missing})")

    if missing > 0:
        raise ValueError(
            f"FATAL: {missing} ECCV caption IDs not found in Karpathy test split. "
            f"Cannot produce valid 5000×25000 matrix. "
            f"Verify dataset_coco.json matches the eccv_caption package version."
        )

    if len(ordered_images) != 5000 or len(ordered_captions) != 25000:
        raise ValueError(
            f"DIMENSION MISMATCH: Expected 5000 images / 25000 captions, "
            f"got {len(ordered_images)} / {len(ordered_captions)}. "
            f"The 5000×25000 contract is non-negotiable."
        )

    return ordered_images, ordered_captions


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_embeddings(model, tokenizer, ordered_images, ordered_captions,
                       transform, max_length, device, batch_size):
    """
    Compute L2-normalized image and text embeddings under no_grad + AMP.

    Returns:
        img_embeds: np.ndarray [N_images, D]
        txt_embeds: np.ndarray [N_captions, D]
        all_iids:   np.ndarray [N_images]
        all_cids:   np.ndarray [N_captions]
    """
    use_amp = device.type == "cuda"

    # --- Image embeddings ---
    logger.info(f"Encoding {len(ordered_images)} images...")
    img_embeds_list = []
    all_iids = []

    for i in tqdm(range(0, len(ordered_images), batch_size), desc="Images"):
        batch_imgs = []
        for j in range(i, min(i + batch_size, len(ordered_images))):
            iid, img_path = ordered_images[j]
            image = Image.open(img_path).convert("RGB")
            batch_imgs.append(transform(image))
            all_iids.append(iid)

        pixel_values = torch.stack(batch_imgs).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            emb = model.encode_image(pixel_values)
        img_embeds_list.append(emb.cpu())

    img_embeds = torch.cat(img_embeds_list, dim=0).numpy()

    # --- Text embeddings ---
    logger.info(f"Encoding {len(ordered_captions)} captions...")
    txt_embeds_list = []
    all_cids = []

    for i in tqdm(range(0, len(ordered_captions), batch_size), desc="Captions"):
        batch_texts = []
        for j in range(i, min(i + batch_size, len(ordered_captions))):
            cid, caption = ordered_captions[j]
            all_cids.append(cid)
            batch_texts.append(caption)

        tok = tokenizer(batch_texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        with torch.amp.autocast("cuda", enabled=use_amp):
            emb = model.encode_text(tok["input_ids"].to(device),
                                    tok["attention_mask"].to(device))
        txt_embeds_list.append(emb.cpu())

    txt_embeds = torch.cat(txt_embeds_list, dim=0).numpy()

    logger.info(f"Embeddings: img={img_embeds.shape}, txt={txt_embeds.shape}")
    return img_embeds, txt_embeds, np.array(all_iids), np.array(all_cids)


# ═══════════════════════════════════════════════════════════════════════════
# Similarity Matrix + Retrieval Dict Construction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_sim_and_retrieval_dicts(img_embeds, txt_embeds, all_iids, all_cids,
                                  device, K=50):
    """
    1. Compute cosine similarity matrix on GPU  (5000 × 25000).
    2. Transfer to CPU as numpy  (.cpu().numpy()).
    3. Build i2t / t2i dicts via descending argsort on numpy.

    Args:
        img_embeds: np.ndarray [N_images, D]  (already L2-normalised)
        txt_embeds: np.ndarray [N_captions, D] (already L2-normalised)
        all_iids:   np.ndarray of image IDs
        all_cids:   np.ndarray of caption IDs
        device:     torch.device
        K:          Top-K results per query (50 sufficient for ECCV max=48)

    Returns:
        i2t: {image_id: [sorted caption_ids by descending similarity]}
        t2i: {caption_id: [sorted image_ids by descending similarity]}
    """
    # Matmul on GPU, then immediately move to numpy to free GPU memory
    img_t = torch.from_numpy(img_embeds).to(device)
    txt_t = torch.from_numpy(txt_embeds).to(device)
    sims = (img_t @ txt_t.T).cpu().numpy()  # [N_images, N_captions]
    del img_t, txt_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Similarity matrix: {sims.shape} → numpy, building i2t/t2i (K={K})...")

    # --- i2t: for each image, rank captions by descending similarity ---
    i2t = {}
    for idx, iid in enumerate(all_iids):
        sorted_indices = np.argsort(-sims[idx])[:K]  # descending
        i2t[int(iid)] = [int(all_cids[i]) for i in sorted_indices]

    # --- t2i: for each caption, rank images by descending similarity ---
    t2i = {}
    for idx, cid in enumerate(all_cids):
        sorted_indices = np.argsort(-sims[:, idx])[:K]  # descending
        t2i[int(cid)] = [int(all_iids[i]) for i in sorted_indices]

    return i2t, t2i


# ═══════════════════════════════════════════════════════════════════════════
# Result Display
# ═══════════════════════════════════════════════════════════════════════════

def print_scores(scores, label):
    """Pretty-print all eccv_caption metrics as an aligned terminal table."""
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"  {label}")
    logger.info("=" * 72)
    logger.info(f"  {'Metric':<28s} {'I2T':>12s}   {'T2I':>12s}")
    logger.info(f"  {'-'*28} {'-'*12}   {'-'*12}")

    for key in sorted(scores.keys()):
        val = scores[key]
        if isinstance(val, dict):
            i2t_s = f"{val.get('i2t', 0):.4f}" if isinstance(val.get("i2t"), float) else "—"
            t2i_s = f"{val.get('t2i', 0):.4f}" if isinstance(val.get("t2i"), float) else "—"
            logger.info(f"  {key:<28s} {i2t_s:>12s}   {t2i_s:>12s}")

    logger.info("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# WandB Hook (optional)
# ═══════════════════════════════════════════════════════════════════════════

def log_to_wandb(scores, config, run_name):
    """Log scores to WandB retrieval-benchmark project as flat metrics + Table."""
    import wandb

    project = config.get("logging", {}).get("wandb_project", "retrieval-benchmark")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    wandb.init(
        project=project,
        name=run_name or f"{job_id}_eccv_eval",
        config={"evaluation": "eccv_caption"},
        job_type="benchmark",
    )

    flat = {}
    for metric_name, dirs in scores.items():
        if isinstance(dirs, dict):
            for d, v in dirs.items():
                flat[f"{metric_name}/{d}"] = v
    wandb.log(flat)

    table = wandb.Table(columns=["metric", "i2t", "t2i"])
    for metric_name, dirs in sorted(scores.items()):
        if isinstance(dirs, dict):
            table.add_data(metric_name,
                           round(dirs.get("i2t", 0), 4),
                           round(dirs.get("t2i", 0), 4))
    wandb.log({"eccv_results": table})

    for k, v in flat.items():
        wandb.run.summary[k] = v

    wandb.finish()
    logger.info(f"Results logged to WandB project '{project}'.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ECCV Caption Benchmark Evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Config path (e.g. configs/config_coco.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help=".pth checkpoint OR HuggingFace model ID for zero-shot")
    parser.add_argument("--run_name", type=str, default=None,
                        help="WandB run name (optional)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding computation (default: 64)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides as key=value")
    args = parser.parse_args()

    # ── Config ──
    config = setup_config(config_path=args.config, overrides=args.override)
    config.setdefault("data", {})
    config["data"].setdefault("max_length", 77)
    config["data"].setdefault("image_size", 336)
    setup_seed(config.get("training", {}).get("seed", 42))

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Model ──
    model = load_model(config, args.checkpoint, device)

    # ── Tokenizer ──
    model_name = config["model"]["image_model_name"]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # ── ECCV caption IDs ──
    logger.info("Loading ECCV Caption protocol...")
    eccv_metric = ECCVMetrics()
    eccv_coco_ids = eccv_metric.coco_ids
    logger.info(f"ECCV canonical caption IDs: {len(eccv_coco_ids)}")

    # ── Data ──
    images_root = config["data"]["images_path"]
    captions_path = config["data"]["captions_path"]
    ordered_images, ordered_captions = load_eccv_test_data(
        captions_path, images_root, eccv_coco_ids
    )

    # ── Embeddings ──
    transform = build_eval_transform(config["data"]["image_size"])
    img_embeds, txt_embeds, all_iids, all_cids = compute_embeddings(
        model, tokenizer, ordered_images, ordered_captions,
        transform, config["data"]["max_length"], device, args.batch_size,
    )

    # ── Similarity matrix → numpy → i2t / t2i ──
    i2t, t2i = build_sim_and_retrieval_dicts(
        img_embeds, txt_embeds, all_iids, all_cids, device
    )

    # ── ECCV metrics ──
    logger.info("Computing eccv_caption metrics...")
    scores = eccv_metric.compute_all_metrics(
        i2t, t2i,
        target_metrics=(
            "eccv_r1", "eccv_map_at_r", "eccv_rprecision",
            "coco_1k_recalls", "coco_5k_recalls", "cxc_recalls",
        ),
        Ks=(1, 5, 10),
        verbose=False,
    )

    # ── Results ──
    label = args.checkpoint or model_name
    print_scores(scores, f"ECCV Benchmark — {label}")

    # ── Optional WandB ──
    use_wandb = config.get("logging", {}).get("use_wandb", True)
    debug_mode = config.get("debug", {}).get("debug_mode", False)
    if use_wandb and not debug_mode:
        log_to_wandb(scores, config, args.run_name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
