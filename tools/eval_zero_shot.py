"""
tools/eval_zero_shot.py
-----------------------
Zero-shot cross-modal retrieval evaluation for B5_mask_only.

Purpose:
    Measure a frozen, pretrained CLIP's retrieval performance on the Karpathy
    test split with two input regimes:

      --input rgb   : original RGB images (reproduces the standard CLIP
                      zero-shot baseline on this dataset).
      --input mask  : colorized SAM segment masks in place of the images, via
                      the SegToRGBRenderer pipeline (deterministic palette,
                      nearest-neighbor resize, no color bleeding at segment
                      boundaries). The model never sees the real pixels.

    The two regimes use IDENTICAL models, transforms (except interpolation
    mode, required for mask-mode fidelity), datasets, and metrics. The only
    controlled difference is what the vision encoder is shown. The delta
    between the two R@K numbers answers: how much structural information,
    independent of appearance, does CLIP's pretrained representation already
    capture?

    No optimizer, no loss, no training loop, no checkpoint is loaded or saved.
    Pretrained weights from HuggingFace only.

Usage:
    python tools/eval_zero_shot.py --config configs/config_flickr30k.yaml --input rgb
    python tools/eval_zero_shot.py --config configs/config_flickr30k.yaml --input mask
    python tools/eval_zero_shot.py --config configs/config_coco.yaml --input rgb
    python tools/eval_zero_shot.py --config configs/config_coco.yaml --input mask

Outputs:
    Prints a formatted R@1/R@5/R@10 table (both i2t and t2i) and optionally
    dumps the same numbers to --output JSON for aggregation.
"""

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# Make `src` importable when the script is invoked from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import create_image_text_dataloader
from src.metrics import compute_recall_at_k
from src.setup import setup_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot CLIP retrieval eval (RGB or mask).")
    p.add_argument("--config", type=str, required=True,
                   help="Path to dataset config (e.g. configs/config_flickr30k.yaml).")
    p.add_argument("--input", type=str, required=True, choices=["rgb", "mask"],
                   help="Input regime. 'rgb' = real images; 'mask' = colorized SAM segment maps.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Evaluation batch size. Independent of the training batch_size in config.")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override dataloader num_workers. Default: config['data']['num_workers'].")
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON path to dump the result table.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device. Default: cuda if available else cpu.")
    return p.parse_args()


@torch.no_grad()
def encode_images(model: CLIPModel, loader, device: torch.device) -> tuple:
    """
    Forward every UNIQUE image in the loader through CLIP's vision tower,
    returning L2-normalized embeddings and the corresponding image IDs in
    loader order. Duplicate image IDs (5 captions per image) are deduplicated
    by keeping the first occurrence.
    """
    seen_ids = set()
    embeddings = []
    unique_ids = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        ids = batch['image_id']  # Tensor or list of ints

        # CLIPModel.get_image_features returns [B, proj_dim] already projected
        # through visual_projection. No extra head, no fine-tuning.
        feats = model.get_image_features(pixel_values=images)  # [B, D]
        feats = F.normalize(feats, dim=-1)

        for i, img_id in enumerate(ids.tolist() if torch.is_tensor(ids) else ids):
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)
            embeddings.append(feats[i].detach().cpu())
            unique_ids.append(img_id)

    if not embeddings:
        raise RuntimeError("Loader yielded zero images; cannot evaluate.")
    img_embeds = torch.stack(embeddings, dim=0)  # [N_unique, D]
    unique_image_ids = torch.tensor(unique_ids, dtype=torch.long)  # [N_unique]
    return img_embeds, unique_image_ids


@torch.no_grad()
def encode_texts(model: CLIPModel, loader, device: torch.device) -> tuple:
    """
    Forward every caption (5 per image in Karpathy splits) through CLIP's
    text tower, returning L2-normalized embeddings and per-caption image IDs
    (for ground-truth matching).
    """
    embeddings = []
    image_ids = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)

        feats = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # [B, D]
        feats = F.normalize(feats, dim=-1)

        embeddings.append(feats.detach().cpu())
        ids = batch['image_id']
        image_ids.extend(ids.tolist() if torch.is_tensor(ids) else ids)

    txt_embeds = torch.cat(embeddings, dim=0)  # [N_captions, D]
    image_ids = torch.tensor(image_ids, dtype=torch.long)  # [N_captions]
    return txt_embeds, image_ids


def format_table(dataset: str, input_mode: str, r_t2i: dict, r_i2t: dict) -> str:
    header = f"=== Zero-shot CLIP — dataset={dataset} | input={input_mode} ==="
    lines = [header]
    lines.append(f"{'Direction':<6} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6}")
    lines.append("-" * 38)
    lines.append(f"{'i2t':<6} | {r_i2t[1]:>6.2f} | {r_i2t[5]:>6.2f} | {r_i2t[10]:>6.2f}")
    lines.append(f"{'t2i':<6} | {r_t2i[1]:>6.2f} | {r_t2i[5]:>6.2f} | {r_t2i[10]:>6.2f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # 1. Load config. Zero-shot eval runs with ZERO registry overrides —
    # this is not a registry variant, it's a measurement tool.
    config = setup_config(config_path=args.config)

    # 2. Flip the data-side switch for mask mode. In RGB mode this key stays
    # false (either missing or from config_base.yaml), so the dataloader
    # behaves exactly like the standard CLIP preprocessing.
    if args.input == "mask":
        config.setdefault('data', {})['use_seg_as_image'] = True
    else:
        # Defensive: make sure no override accidentally flipped it on.
        config.setdefault('data', {})['use_seg_as_image'] = False

    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers

    # The dataloader factory reads batch_size from config for train split;
    # for eval it uses the same key. Override it with our CLI-level value so
    # eval batching is independent of training batch size.
    config.setdefault('training', {})['batch_size'] = args.batch_size

    # 3. Device.
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # 4. Pretrained CLIP — raw from HuggingFace. No DualEncoder wrapper, no
    # freezing strategy, no logit_scale reset, no projection heads beyond
    # CLIP's native visual_projection / text_projection.
    model_name = config['model']['image_model_name']
    logger.info(f"Loading pretrained CLIP: {model_name}")
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    # 5. Tokenizer and test-split dataloader. The factory builds the
    # deterministic eval transform (Resize+CenterCrop+Normalize) and, when
    # data.use_seg_as_image is true, the SegToRGBRenderer with NEAREST resize.
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    test_loader = create_image_text_dataloader(config, tokenizer, split="test")
    logger.info(f"Test loader ready: {len(test_loader.dataset)} caption samples")

    # 6. Encode images (unique) and captions.
    logger.info("Encoding images ...")
    img_embeds, unique_image_ids = encode_images(model, test_loader, device)
    logger.info(f"  -> {img_embeds.shape[0]} unique images, dim={img_embeds.shape[1]}")

    logger.info("Encoding captions ...")
    txt_embeds, image_ids = encode_texts(model, test_loader, device)
    logger.info(f"  -> {txt_embeds.shape[0]} captions")

    # 7. Retrieval metrics on a single similarity matrix.
    logger.info("Computing Recall@{1,5,10} ...")
    r_t2i, r_i2t = compute_recall_at_k(
        img_embeds=img_embeds,
        txt_embeds=txt_embeds,
        image_ids=image_ids,
        unique_image_ids=unique_image_ids,
        k_values=[1, 5, 10],
    )

    dataset = config['data']['dataset']
    print("\n" + format_table(dataset, args.input, r_t2i, r_i2t) + "\n")

    if args.output:
        payload = OrderedDict([
            ("dataset", dataset),
            ("input", args.input),
            ("model", model_name),
            ("split", "test"),
            ("n_unique_images", int(img_embeds.shape[0])),
            ("n_captions", int(txt_embeds.shape[0])),
            ("recall_i2t", {str(k): r_i2t[k] for k in sorted(r_i2t)}),
            ("recall_t2i", {str(k): r_t2i[k] for k in sorted(r_t2i)}),
            ("palette_size", config.get('data', {}).get('seg_palette_size')),
            ("seg_map_dir", config.get('data', {}).get('seg_map_dir')),
            ("use_seg_as_image", config.get('data', {}).get('use_seg_as_image', False)),
        ])
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
