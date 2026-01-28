#!/usr/bin/env python3
"""
diagnose_model.py
-----------------
Comprehensive Failure Analysis & Visualization for Cross-Modal Retrieval.

Features:
- Robust data loading for COCO & Flickr30k (Karpathy JSON via CaptionImageDataset)
- Supports both Text-to-Image (T2I) and Image-to-Text (I2T) retrieval
- Top-5 visualization for each failure (not only Top-1)
- Computes 4 diagnostic scores per failure:
    * Score_Conf  : model similarity for retrieved Top-1 item
    * Score_GT    : model similarity for the correct ground-truth item
    * Score_Vis   : visual confusion (GT image vs retrieved image)
    * Score_Sem   : semantic validity (query caption vs retrieved image captions)
- Aggregates statistics over all failures
- Generates a standalone HTML report (`failure_report.html`)
"""

import argparse
import yaml
import torch
import json
import os
import sys
import logging
import base64
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DualEncoder
from src.data import get_dataloader, CaptionImageDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_similarity(embed1: torch.Tensor, embed2: torch.Tensor) -> float:
    """Cosine similarity between two 1D or 2D normalized embeddings."""
    if embed1.ndim == 1:
        embed1 = embed1.unsqueeze(0)
    if embed2.ndim == 1:
        embed2 = embed2.unsqueeze(0)
    return F.cosine_similarity(embed1, embed2, dim=-1).item()


def build_metadata(dataset: CaptionImageDataset):
    """
    Build robust metadata mappings from CaptionImageDataset for both COCO & Flickr30k.

    Returns:
        image_ids: List[int]                - unique image_ids sorted
        image_id_to_index: Dict[int, int]   - image_id -> image index (0..M-1)
        image_id_to_captions: Dict[int, List[str]]
        image_id_to_path: Dict[int, str]    - full image path
        caption_index_to_image_id: List[int] - length N (dataset size)
    """
    images_root = dataset.images_root_path

    image_id_to_captions = {}
    image_id_to_fileinfo = {}
    caption_index_to_image_id = []

    for idx, sample in enumerate(dataset.samples):
        img_id = sample["image_id"]
        caption = sample["caption"]
        filepath = sample.get("filepath", "").strip()
        filename = sample["filename"]

        # Captions
        image_id_to_captions.setdefault(img_id, []).append(caption)

        # File info (keep first occurrence)
        if img_id not in image_id_to_fileinfo:
            image_id_to_fileinfo[img_id] = (filepath, filename)

        caption_index_to_image_id.append(img_id)

    # Build image paths using same logic as __getitem__
    image_id_to_path = {}
    for img_id, (filepath, filename) in image_id_to_fileinfo.items():
        if filepath:
            image_path = os.path.join(images_root, filepath, filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(images_root, filename)
        else:
            image_path = os.path.join(images_root, filename)
        image_id_to_path[img_id] = image_path

    image_ids = sorted(image_id_to_captions.keys())
    image_id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}

    return (
        image_ids,
        image_id_to_index,
        image_id_to_captions,
        image_id_to_path,
        caption_index_to_image_id,
    )


def encode_embeddings(model, dataloader, device):
    """
    Encode all images and captions in the provided dataloader.

    Returns:
        text_embeds: [N_captions, D]       (CPU, L2-normalized)
        image_embeds: [M_images, D]       (CPU, L2-normalized, one per image_id)
        image_ids: List[int]              (length M, sorted)
        caption_index_to_image_index: List[int] (len N, maps caption idx -> image index)
    """
    dataset: CaptionImageDataset = dataloader.dataset
    n_samples = len(dataset)

    (
        image_ids,
        image_id_to_index,
        image_id_to_captions,
        image_id_to_path,
        caption_index_to_image_id,
    ) = build_metadata(dataset)

    logger.info(f"Dataset: {n_samples} captions, {len(image_ids)} unique images")

    text_embeds = None
    # For images: accumulate per image_id then normalize
    image_sum = {}
    image_count = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["index"].cpu().numpy()
            batch_image_ids = batch["image_id"]
            if isinstance(batch_image_ids, torch.Tensor):
                batch_image_ids = batch_image_ids.cpu().numpy()

            img_embeds, txt_embeds = model(images, input_ids, attention_mask)

            # Embeddings from model are already normalized, but be safe
            img_embeds = F.normalize(img_embeds, dim=-1)
            txt_embeds = F.normalize(txt_embeds, dim=-1)

            if text_embeds is None:
                embed_dim = txt_embeds.size(-1)
                text_embeds = torch.zeros(n_samples, embed_dim, dtype=torch.float32)

            text_embeds[indices] = txt_embeds.cpu()

            # Accumulate per image_id
            for j, idx in enumerate(indices):
                img_id = int(batch_image_ids[j])
                vec = img_embeds[j].cpu()
                if img_id not in image_sum:
                    image_sum[img_id] = vec.clone()
                    image_count[img_id] = 1
                else:
                    image_sum[img_id] += vec
                    image_count[img_id] += 1

    # Build image embedding matrix (one per image_id)
    embed_dim = text_embeds.size(-1)
    image_embeds = torch.zeros(len(image_ids), embed_dim, dtype=torch.float32)
    for img_id in image_ids:
        summed = image_sum[img_id] / max(image_count.get(img_id, 1), 1)
        image_embeds[image_id_to_index[img_id]] = F.normalize(summed, dim=-1)

    # Map caption index -> image index
    caption_index_to_image_index = [
        image_id_to_index[int(img_id)] for img_id in caption_index_to_image_id
    ]

    return (
        text_embeds,
        image_embeds,
        image_ids,
        caption_index_to_image_index,
        image_id_to_captions,
        image_id_to_path,
    )


def analyze_t2i_failures(
    text_embeds,
    image_embeds,
    image_ids,
    caption_index_to_image_index,
    image_id_to_captions,
    image_id_to_path,
    dataset: CaptionImageDataset,
    max_failures: int = 200,
):
    """
    Text-to-Image failure analysis.

    For each caption query (text), retrieve top-5 images and analyze R@1 failures.
    """
    sim_matrix = text_embeds @ image_embeds.T  # [N_captions, M_images]

    failures = []
    n_queries = text_embeds.size(0)

    for q_idx in tqdm(range(n_queries), desc="Analyzing T2I failures"):
        scores = sim_matrix[q_idx]
        top_scores, top_indices = torch.topk(scores, k=min(5, scores.size(0)))

        gt_img_index = caption_index_to_image_index[q_idx]
        gt_img_id = image_ids[gt_img_index]

        pred_img_index = top_indices[0].item()
        pred_img_id = image_ids[pred_img_index]

        # R@1 success -> skip
        if pred_img_index == gt_img_index:
            continue

        # --- FAILURE ---
        query_caption = dataset.samples[q_idx]["caption"]
        gt_image_path = image_id_to_path[gt_img_id]

        # Top-5 retrieved images metadata
        top5 = []
        for rank, (idx, sc) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
            img_id = image_ids[idx]
            top5.append(
                {
                    "rank": rank,
                    "image_id": int(img_id),
                    "image_path": image_id_to_path[img_id],
                    "caption_example": image_id_to_captions[img_id][0]
                    if image_id_to_captions.get(img_id)
                    else "",
                    "score_model": round(float(sc), 4),
                    "is_gt": bool(idx == gt_img_index),
                }
            )

        # Scores (Top-1 false positive vs GT)
        score_conf = float(top_scores[0].item())
        score_gt = float(scores[gt_img_index].item())

        gt_img_embed = image_embeds[gt_img_index]
        pred_img_embed = image_embeds[pred_img_index]
        score_vis = compute_similarity(gt_img_embed, pred_img_embed)

        # Semantic Validity: query caption vs all captions of retrieved image
        query_embed = text_embeds[q_idx]  # already normalized
        retrieved_caps_indices = [
            i for i, s in enumerate(dataset.samples) if s["image_id"] == pred_img_id
        ]
        if retrieved_caps_indices:
            retrieved_cap_embeds = text_embeds[retrieved_caps_indices]  # [K, D]
            sem_sims = torch.matmul(retrieved_cap_embeds, query_embed.unsqueeze(-1)).squeeze(-1)
            score_sem = float(sem_sims.max().item())
        else:
            score_sem = 0.0

        failures.append(
            {
                "direction": "t2i",
                "query_index": int(q_idx),
                "query_caption": query_caption,
                "gt_image_id": int(gt_img_id),
                "gt_image_path": gt_image_path,
                "top5": top5,
                "scores": {
                    "confidence": round(score_conf, 4),
                    "gt_score": round(score_gt, 4),
                    "visual_sim": round(score_vis, 4),
                    "semantic_validity": round(score_sem, 4),
                },
            }
        )

        if len(failures) >= max_failures:
            break

    return failures


def analyze_i2t_failures(
    text_embeds,
    image_embeds,
    image_ids,
    image_id_to_captions,
    image_id_to_path,
    dataset: CaptionImageDataset,
    max_failures: int = 200,
):
    """
    Image-to-Text failure analysis.

    For each image query, retrieve top-5 captions and analyze R@1 failures.
    """
    # Build mapping: image_id -> list of caption indices
    image_id_to_caption_indices = {}
    for idx, sample in enumerate(dataset.samples):
        img_id = sample["image_id"]
        image_id_to_caption_indices.setdefault(img_id, []).append(idx)

    # One query per unique image (Image-to-Text)
    sim_matrix = image_embeds @ text_embeds.T  # [M_images, N_captions]

    failures = []
    n_images = image_embeds.size(0)

    for img_index in tqdm(range(n_images), desc="Analyzing I2T failures"):
        img_id = image_ids[img_index]
        scores = sim_matrix[img_index]
        top_scores, top_indices = torch.topk(scores, k=min(5, scores.size(0)))

        gt_caption_indices = image_id_to_caption_indices.get(img_id, [])
        if not gt_caption_indices:
            continue

        pred_caption_index = top_indices[0].item()

        # Success if top-1 caption belongs to same image
        if pred_caption_index in gt_caption_indices:
            continue

        # --- FAILURE ---
        gt_image_path = image_id_to_path[img_id]
        gt_captions = image_id_to_captions[img_id]

        # Top-5 retrieved captions
        top5 = []
        for rank, (c_idx, sc) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
            sample = dataset.samples[c_idx]
            cid = sample["image_id"]
            top5.append(
                {
                    "rank": rank,
                    "caption_index": int(c_idx),
                    "caption": sample["caption"],
                    "image_id": int(cid),
                    "image_path": image_id_to_path.get(cid, ""),
                    "score_model": round(float(sc), 4),
                    "is_gt": bool(cid == img_id),
                }
            )

        # Scores (Top-1 false positive vs GT)
        score_conf = float(top_scores[0].item())
        # Best GT caption similarity (any of the GT captions)
        gt_scores = scores[gt_caption_indices]
        score_gt = float(gt_scores.max().item())

        gt_img_embed = image_embeds[img_index]
        pred_img_id = dataset.samples[pred_caption_index]["image_id"]
        pred_img_index = image_ids.index(pred_img_id)
        pred_img_embed = image_embeds[pred_img_index]
        score_vis = compute_similarity(gt_img_embed, pred_img_embed)

        # Semantic Validity: pick a representative GT caption as query text
        query_caption_index = gt_caption_indices[0]
        query_embed = text_embeds[query_caption_index]

        retrieved_caps_indices = image_id_to_caption_indices.get(pred_img_id, [])
        if retrieved_caps_indices:
            retrieved_cap_embeds = text_embeds[retrieved_caps_indices]
            sem_sims = torch.matmul(retrieved_cap_embeds, query_embed.unsqueeze(-1)).squeeze(-1)
            score_sem = float(sem_sims.max().item())
        else:
            score_sem = 0.0

        failures.append(
            {
                "direction": "i2t",
                "query_image_id": int(img_id),
                "query_image_path": gt_image_path,
                "gt_captions": gt_captions,
                "top5": top5,
                "scores": {
                    "confidence": round(score_conf, 4),
                    "gt_score": round(score_gt, 4),
                    "visual_sim": round(score_vis, 4),
                    "semantic_validity": round(score_sem, 4),
                },
            }
        )

        if len(failures) >= max_failures:
            break

    return failures


def aggregate_statistics(failures):
    """Compute aggregate semantic & visual confusion statistics."""
    if not failures:
        return {}

    n = len(failures)
    sem_high = sum(1 for f in failures if f["scores"]["semantic_validity"] > 0.8)
    vis_high = sum(1 for f in failures if f["scores"]["visual_sim"] > 0.9)

    stats = {
        "num_failures": n,
        "semantic_high_count": sem_high,
        "semantic_high_pct": 100.0 * sem_high / n,
        "visual_high_count": vis_high,
        "visual_high_pct": 100.0 * vis_high / n,
    }

    logger.info(
        f'Out of {n} failures, {stats["semantic_high_pct"]:.2f}% have a Semantic Score > 0.8 '
        f'({sem_high}/{n}).'
    )
    logger.info(
        f'Out of {n} failures, {stats["visual_high_pct"]:.2f}% have a Visual Score > 0.9 '
        f'({vis_high}/{n}).'
    )

    return stats


def generate_html_report(all_failures, output_html):
    """Generate a standalone HTML report for all failures.

    Images are embedded as base64 data URIs so that the HTML file
    is fully self-contained and portable across machines.
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    def image_to_base64(img_path: str) -> str:
        """Encode image file as base64 string. Returns empty string on failure."""
        if not img_path:
            return ""
        try:
            with open(img_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as exc:
            logger.warning(f"Failed to read image for HTML report: {img_path} ({exc})")
            return ""

    def img_tag(img_path: str, extra_style: str = "") -> str:
        """Return an <img> tag with base64-embedded image, or a placeholder if missing."""
        b64 = image_to_base64(img_path)
        if not b64:
            return '<div class="missing-img">Image not found</div>'
        # Default to jpeg; for Flickr/COCO this is usually correct.
        style_attr = f' style="{extra_style}"' if extra_style else ""
        return f'<img src="data:image/jpeg;base64,{b64}" class="thumb"{style_attr}>'

    def scores_html(scores):
        return (
            f'Conf: {scores["confidence"]:.3f}<br>'
            f'GT: {scores["gt_score"]:.3f}<br>'
            f'Vis: {scores["visual_sim"]:.3f}<br>'
            f'Sem: {scores["semantic_validity"]:.3f}'
        )

    rows = []
    for case in all_failures:
        direction = case["direction"]

        # Column 1: Query
        if direction == "t2i":
            query_html = f'<div class="query-text">{case["query_caption"]}</div>'
        else:
            query_img_tag = img_tag(case["query_image_path"])
            query_html = (
                f"{query_img_tag}"
                f'<div class="caption-list">{"<br>".join(case["gt_captions"])}</div>'
            )

        # Column 2: Ground Truth
        if direction == "t2i":
            gt_html = img_tag(case["gt_image_path"])
        else:
            gt_html = '<div class="gt-text">' + "<br>".join(case["gt_captions"]) + "</div>"

        # Columns 3-7: Top-5 retrieved
        retrieved_cells = []
        for item in case["top5"]:
            border_color = "green" if item["is_gt"] else "red"
            if direction == "t2i":
                img_html = img_tag(
                    item["image_path"], extra_style=f"border: 3px solid {border_color};"
                )
                cell_content = (
                    f"{img_html}"
                    f'<div class="score">Score: {item["score_model"]:.3f}</div>'
                    f'<div class="caption">{item.get("caption_example","")}</div>'
                )
            else:
                cell_content = (
                    f'<div class="retrieved-caption" '
                    f'style="border: 3px solid {border_color};">'
                    f'{item["caption"]}</div>'
                    f'<div class="score">Score: {item["score_model"]:.3f}</div>'
                )
            retrieved_cells.append(f"<td>{cell_content}</td>")

        row_html = (
            "<tr>"
            f'<td class="query-col">{query_html}</td>'
            f'<td class="gt-col">{gt_html}</td>'
            + "".join(retrieved_cells)
            + f'<td class="scores-col">{scores_html(case["scores"])}</td>'
            + "</tr>"
        )
        rows.append(row_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Failure Analysis Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      background-color: #fafafa;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      vertical-align: top;
    }}
    th {{
      background-color: #f2f2f2;
      text-align: center;
    }}
    .thumb {{
      max-width: 200px;
      max-height: 200px;
      display: block;
      margin-bottom: 4px;
    }}
    .query-text {{
      font-weight: bold;
    }}
    .gt-text {{
      font-weight: bold;
      color: #2e7d32;
    }}
    .retrieved-caption {{
      padding: 4px;
    }}
    .score {{
      font-size: 0.85em;
      color: #555;
      margin-top: 4px;
    }}
    .scores-col {{
      font-size: 0.85em;
      width: 120px;
    }}
    .caption-list {{
      margin-top: 6px;
      font-size: 0.9em;
    }}
  </style>
</head>
<body>
  <h2>Failure Analysis Report</h2>
  <p>Total failures visualized: {len(all_failures)}</p>
  <table>
    <tr>
      <th>Query (Text / Image)</th>
      <th>Ground Truth (Image / Captions)</th>
      <th colspan="5">Top-5 Retrieved Items</th>
      <th>Scores (Top-1)</th>
    </tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML failure report saved to: {output_html}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Override image root for local execution (config.data.images_path)",
    )
    parser.add_argument(
        "--captions_path",
        type=str,
        default=None,
        help="Override captions JSON path for local execution (config.data.captions_path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diagnosis",
        help="Directory to save JSON + HTML reports",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to analyze",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["t2i", "i2t", "both"],
        help="Retrieval direction to analyze",
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=200,
        help="Maximum number of failures to store/visualize per direction",
    )
    args = parser.parse_args()

    # 1. Load Config & Override Paths
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.image_root:
        logger.info(f"Overriding config['data']['images_path'] to: {args.image_root}")
        if "data" not in config:
            config["data"] = {}
        config["data"]["images_path"] = args.image_root

    if args.captions_path:
        logger.info(
            f"Overriding config['data']['captions_path'] to: {args.captions_path}"
        )
        if "data" not in config:
            config["data"] = {}
        config["data"]["captions_path"] = args.captions_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running failure analysis on device: {device}")

    # 2. Tokenizer, DataLoader, Model
    clip_model_name = config["model"]["image_model_name"]
    logger.info(f"Loading CLIP Tokenizer: {clip_model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    logger.info(f"Loading {args.split} split...")
    loader = get_dataloader(config, tokenizer, split=args.split)
    dataset: CaptionImageDataset = loader.dataset

    logger.info(f"Building DualEncoder model and loading checkpoint: {args.checkpoint}")
    model = DualEncoder(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Encode all embeddings (batched, CPU storage to avoid GPU OOM)
    (
        text_embeds,
        image_embeds,
        image_ids,
        caption_index_to_image_index,
        image_id_to_captions,
        image_id_to_path,
    ) = encode_embeddings(model, loader, device)

    # 4. Analyze failures for requested directions
    all_failures = []

    if args.direction in ["t2i", "both"]:
        t2i_failures = analyze_t2i_failures(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            image_ids=image_ids,
            caption_index_to_image_index=caption_index_to_image_index,
            image_id_to_captions=image_id_to_captions,
            image_id_to_path=image_id_to_path,
            dataset=dataset,
            max_failures=args.max_failures,
        )
        logger.info(f"T2I failures collected: {len(t2i_failures)}")
        all_failures.extend(t2i_failures)

    if args.direction in ["i2t", "both"]:
        i2t_failures = analyze_i2t_failures(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            image_ids=image_ids,
            image_id_to_captions=image_id_to_captions,
            image_id_to_path=image_id_to_path,
            dataset=dataset,
            max_failures=args.max_failures,
        )
        logger.info(f"I2T failures collected: {len(i2t_failures)}")
        all_failures.extend(i2t_failures)

    # 5. Aggregate statistics (over all collected failures)
    stats = aggregate_statistics(all_failures)

    # 6. Save JSON + HTML
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "failure_report.json")
    html_path = os.path.join(args.output_dir, "failure_report.html")

    output_payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "direction": args.direction,
        "split": args.split,
        "failures": all_failures,
        "stats": stats,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON failure report saved to: {json_path}")

    generate_html_report(all_failures, html_path)


if __name__ == "__main__":
    main()