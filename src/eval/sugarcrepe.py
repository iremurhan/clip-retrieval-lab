"""
src/eval/sugarcrepe.py
----------------------
SugarCrepe (NeurIPS'23) compositional understanding evaluation core.

The shared, model-callable core for SugarCrepe metrics. Takes an in-memory
DualEncoder (no checkpoint I/O) and returns a dict of per-subcategory
accuracies plus a macro_avg.

Used by:
  - scripts/eval/eval_sugarcrepe.py (standalone CLI; loads a checkpoint then calls in)
  - src/train.py::Trainer.evaluate (in-pipeline hook; future, Task 2)
"""

import json
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

SUBCATEGORIES = [
    "add_att", "add_obj",
    "replace_att", "replace_obj", "replace_rel",
    "swap_att", "swap_obj",
]


def evaluate_subcategory(model, tokenizer, transform, data, images_dir, max_length, device):
    """
    Evaluate a single SugarCrepe sub-category.

    Args:
        model: DualEncoder in eval mode.
        tokenizer: CLIPTokenizer.
        transform: callable PIL.Image -> Tensor [3, H, W].
        data: dict keyed by string indices, each with 'filename', 'caption',
            'negative_caption'.
        images_dir: directory containing the referenced images.
        max_length: tokenizer max_length.
        device: torch.device.

    Returns:
        accuracy (float): fraction of triplets with sim(image, pos) > sim(image, neg).
    """
    correct = 0
    total = 0

    for entry in data.values():
        filename = entry["filename"]
        pos_caption = entry["caption"]
        neg_caption = entry["negative_caption"]

        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            # Official COCO 2017 images use bare IDs, but keep compatibility
            # with older local COCO 2014 layouts that may use split prefixes.
            prefixed_paths = [
                os.path.join(images_dir, f"COCO_val2017_{filename}"),
                os.path.join(images_dir, f"COCO_val2014_{filename}"),
            ]
            img_path = next((p for p in prefixed_paths if os.path.isfile(p)), None)
            if img_path is None:
                logger.warning(
                    f"Image not found, skipping: {os.path.join(images_dir, filename)}"
                )
                continue

        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

        tok_pos = tokenizer(
            pos_caption, return_tensors="pt",
            max_length=max_length, padding="max_length", truncation=True,
        )
        tok_neg = tokenizer(
            neg_caption, return_tensors="pt",
            max_length=max_length, padding="max_length", truncation=True,
        )

        pos_ids = tok_pos["input_ids"].to(device)            # [1, L]
        pos_mask = tok_pos["attention_mask"].to(device)      # [1, L]
        neg_ids = tok_neg["input_ids"].to(device)            # [1, L]
        neg_mask = tok_neg["attention_mask"].to(device)      # [1, L]

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            img_emb = model.encode_image(img_tensor)                   # [1, D]
            pos_emb = model.encode_text(pos_ids, pos_mask)             # [1, D]
            neg_emb = model.encode_text(neg_ids, neg_mask)             # [1, D]

        sim_pos = F.cosine_similarity(img_emb, pos_emb).item()
        sim_neg = F.cosine_similarity(img_emb, neg_emb).item()

        if sim_pos > sim_neg:
            correct += 1
        total += 1

    if total == 0:
        raise RuntimeError("No valid samples found. Check images_dir and data files.")
    return correct / total


def evaluate_sugarcrepe(
    model,
    tokenizer,
    transform,
    device,
    data_dir,
    images_dir,
    max_length=77,
    splits=("replace", "swap", "add"),
):
    """
    Evaluate SugarCrepe compositional understanding metrics on an in-memory model.

    Iterates over the SugarCrepe subcategories whose names start with any of
    `splits` (parent-level filter, e.g. "replace" -> {replace_att, replace_obj,
    replace_rel}). For each subcategory, computes the fraction of triplets
    (image, pos_caption, neg_caption) where sim(image, pos) > sim(image, neg).

    Args:
        model: DualEncoder in eval mode (caller's responsibility — this function
            does not toggle model.train/eval).
        tokenizer: CLIPTokenizer matching the model.
        transform: callable PIL.Image -> Tensor [3, H, W] (e.g. build_eval_transform).
        device: torch.device on which the model lives.
        data_dir: directory containing {subcategory}.json files.
        images_dir: directory containing the referenced images (typically
            datasets/coco/val2017).
        max_length: tokenizer max_length, default 77 (CLIP standard).
        splits: tuple of parent-category prefixes to evaluate. Default covers all.

    Returns:
        dict[str, float] with keys equal to the matched SUBCATEGORIES plus
        "macro_avg" (mean over those keys). All values in [0, 1].
    """
    selected = [s for s in SUBCATEGORIES if any(s.startswith(p) for p in splits)]
    if not selected:
        raise ValueError(
            f"No SugarCrepe subcategories match splits={splits}. "
            f"Valid prefixes: {sorted({c.split('_')[0] for c in SUBCATEGORIES})}"
        )

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    results = {}
    for subcat in selected:
        json_path = os.path.join(data_dir, f"{subcat}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"SugarCrepe data file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        acc = evaluate_subcategory(
            model, tokenizer, transform, data, images_dir, max_length, device,
        )
        results[subcat] = acc
        logger.info(f"  sugarcrepe/{subcat}: {acc:.4f} ({len(data)} samples)")

    results["macro_avg"] = sum(results.values()) / len(results)
    logger.info(f"  sugarcrepe/macro_avg: {results['macro_avg']:.4f}")
    return results
