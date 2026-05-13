"""
MMVP-VLM forced-choice evaluation core.

The benchmark is organized as 135 image-pair/caption-pair decisions across
9 visual patterns. For each pair, both captions must select their matching
image: caption 1 should prefer image 1, and caption 2 should prefer image 2.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image


logger = logging.getLogger(__name__)

PATTERN_LABELS = {
    "orientation": "orientation",
    "orientation and direction": "orientation",
    "presence": "presence",
    "presence of specific features": "presence",
    "state": "state",
    "state and condition": "state",
    "quantity": "quantity",
    "quantity and count": "quantity",
    "spatial": "spatial",
    "positional and relational context": "spatial",
    "color": "color",
    "color and appearance": "color",
    "structural": "structural",
    "structural character": "structural",
    "structural characteristics": "structural",
    "text": "text_rendering",
    "texts": "text_rendering",
    "text rendering": "text_rendering",
    "camera perspective": "viewpoint",
    "viewpoint": "viewpoint",
    "viewpoint and perspective": "viewpoint",
}

PATTERN_ORDER = [
    "orientation",
    "presence",
    "state",
    "quantity",
    "spatial",
    "color",
    "structural",
    "text_rendering",
    "viewpoint",
]


@dataclass(frozen=True)
class MMVPPair:
    pair_id: int
    pattern: str
    image1: Path
    image2: Path
    caption1: str
    caption2: str
    qid1: int
    qid2: int


def _norm_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _pick_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    by_norm = {_norm_key(name): name for name in fieldnames}
    for candidate in candidates:
        key = _norm_key(candidate)
        if key in by_norm:
            return by_norm[key]
    raise KeyError(f"Could not find any of columns {list(candidates)} in {list(fieldnames)}")


def _normalize_pattern(value: object) -> str:
    text = str(value).strip().replace("_", " ")
    return PATTERN_LABELS.get(text.lower(), text.lower().replace(" ", "_"))


def _pattern_dir_names(pattern: str) -> list[str]:
    return {
        "orientation": ["Orientation"],
        "presence": ["Presence"],
        "state": ["State"],
        "quantity": ["Quantity"],
        "spatial": ["Spatial", "Positional"],
        "color": ["Color"],
        "structural": ["Structural Character", "Structural", "Structural_Character"],
        "text_rendering": ["Text", "Texts"],
        "viewpoint": ["Camera_Perspective", "Camera Perspective", "Viewpoint"],
    }.get(pattern, [pattern])


def _find_questions_csv(data_dir: Path) -> Path:
    candidates = [data_dir / "Questions.csv", data_dir / "questions.csv"]
    candidates.extend(data_dir.glob("**/Questions.csv"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find Questions.csv under {data_dir}")


def _find_images_root(data_dir: Path) -> Path:
    candidates = [
        data_dir / "MLLM_VLM Images",
        data_dir / "MLLM_VLM_Images",
        data_dir / "MMVP_VLM_Images",
        data_dir / "MMVP VLM Images",
        data_dir / "images",
        data_dir,
    ]
    for path in candidates:
        if path.is_dir() and any(
            (path / dirname).is_dir() for pattern in PATTERN_ORDER for dirname in _pattern_dir_names(pattern)
        ):
            return path
    raise FileNotFoundError(f"Could not find MMVP-VLM image folders under {data_dir}")


def _image_path(images_root: Path, pattern: str, qid: int, row: dict) -> Path:
    path_cols = ["image", "image_path", "filename", "file_name", "img", "img_path"]
    for col in path_cols:
        if col in row and str(row[col]).strip():
            candidate = Path(str(row[col]).strip())
            if candidate.is_absolute() and candidate.is_file():
                return candidate
            rooted = images_root / candidate
            if rooted.is_file():
                return rooted

    tried = []
    for dirname in _pattern_dir_names(pattern):
        pattern_dir = images_root / dirname
        direct = pattern_dir / f"{qid}.jpg"
        tried.append(direct)
        if direct.is_file():
            return direct
    raise FileNotFoundError(
        f"Could not resolve MMVP image for qid={qid}, pattern={pattern}. Tried: {', '.join(str(p) for p in tried)}"
    )


def load_mmvp_pairs(data_dir: str | Path) -> list[MMVPPair]:
    """Load official MMVP-VLM Questions.csv into paired evaluation items."""
    data_dir = Path(data_dir)
    questions_csv = _find_questions_csv(data_dir)
    images_root = _find_images_root(data_dir)
    with questions_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Questions.csv is empty: {questions_csv}")

    fieldnames = rows[0].keys()
    qid_col = _pick_column(fieldnames, ["qid", "question_id", "question id", "id", "index"])
    pattern_col = _pick_column(fieldnames, ["qtype", "type", "category", "pattern", "label"])
    caption_col = _pick_column(fieldnames, ["statement", "text", "caption", "question", "prompt"])

    parsed = []
    for row in rows:
        qid = int(str(row[qid_col]).strip())
        pattern = _normalize_pattern(row[pattern_col])
        parsed.append((qid, pattern, str(row[caption_col]).strip(), row))
    parsed.sort(key=lambda item: item[0])

    if len(parsed) % 2 != 0:
        raise ValueError(f"Expected an even number of MMVP-VLM rows, got {len(parsed)}")

    pairs = []
    for pair_idx, offset in enumerate(range(0, len(parsed), 2), start=1):
        qid1, pattern1, caption1, row1 = parsed[offset]
        qid2, pattern2, caption2, row2 = parsed[offset + 1]
        if pattern1 != pattern2:
            raise ValueError(f"MMVP pair has mismatched patterns: qid {qid1}={pattern1}, qid {qid2}={pattern2}")
        pairs.append(
            MMVPPair(
                pair_id=pair_idx,
                pattern=pattern1,
                image1=_image_path(images_root, pattern1, qid1, row1),
                image2=_image_path(images_root, pattern2, qid2, row2),
                caption1=caption1,
                caption2=caption2,
                qid1=qid1,
                qid2=qid2,
            )
        )

    logger.info("Loaded %d MMVP-VLM pairs from %s", len(pairs), questions_csv)
    return pairs


@torch.no_grad()
def evaluate_pair(model, tokenizer, transform, pair: MMVPPair, max_length: int, device: torch.device) -> dict:
    images = [
        transform(Image.open(pair.image1).convert("RGB")),
        transform(Image.open(pair.image2).convert("RGB")),
    ]
    image_tensor = torch.stack(images, dim=0).to(device)
    tokenized = tokenizer(
        [pair.caption1, pair.caption2],
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        image_embeds = model.encode_image(image_tensor)
        text_embeds = model.encode_text(input_ids, attention_mask)
    image_embeds = F.normalize(image_embeds.float(), p=2, dim=-1)
    text_embeds = F.normalize(text_embeds.float(), p=2, dim=-1)
    sims = image_embeds @ text_embeds.t()

    pred1 = int(torch.argmax(sims[:, 0]).item())
    pred2 = int(torch.argmax(sims[:, 1]).item())
    correct = pred1 == 0 and pred2 == 1
    diagonal_margin = float((sims[0, 0] + sims[1, 1] - sims[0, 1] - sims[1, 0]).item())
    return {
        "pair_id": pair.pair_id,
        "pattern": pair.pattern,
        "qid1": pair.qid1,
        "qid2": pair.qid2,
        "pred1": pred1 + 1,
        "pred2": pred2 + 1,
        "correct": bool(correct),
        "diagonal_margin": diagonal_margin,
        "sim_11": float(sims[0, 0].item()),
        "sim_12": float(sims[0, 1].item()),
        "sim_21": float(sims[1, 0].item()),
        "sim_22": float(sims[1, 1].item()),
    }


def summarize_results(pair_results: list[dict]) -> dict:
    summary = {}
    for pattern in PATTERN_ORDER:
        rows = [row for row in pair_results if row["pattern"] == pattern]
        if rows:
            summary[pattern] = sum(row["correct"] for row in rows) / len(rows)
    if not summary:
        raise RuntimeError("No MMVP-VLM results to summarize.")
    summary["overall"] = sum(summary.values()) / len(summary)
    return summary


def evaluate_mmvp_vlm(
    model,
    tokenizer,
    transform,
    device,
    data_dir,
    max_length=77,
) -> tuple[dict, list[dict]]:
    pairs = load_mmvp_pairs(data_dir)
    results = [
        evaluate_pair(model, tokenizer, transform, pair, max_length=max_length, device=device)
        for pair in pairs
    ]
    summary = summarize_results(results)
    for key, value in summary.items():
        logger.info("  mmvp_vlm/%s: %.4f", key, value)
    return summary, results
