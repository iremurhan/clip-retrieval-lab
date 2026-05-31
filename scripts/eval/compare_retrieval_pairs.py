"""Paired baseline-vs-intervention retrieval comparison.

Loads two checkpoints (a baseline and an intervention), computes ground-truth ranks for
every query on the test set, and surfaces the cases that *flipped*:

  fixed   — the baseline ranked the ground truth poorly (rank > threshold) but the
            intervention ranks it at top-1 (rank 0): the intervention fixed it.
  broken  — the baseline had it right (rank 0) but the intervention pushed it down:
            a regression introduced by the intervention.

Both directions (I2T, T2I) are reported. Outputs a self-contained HTML gallery (query +
top-k retrieved, side by side for both models) plus a JSON sidecar.

Reuses the model/embedding/rank machinery in extract_failures.py so the ranking matches the
trainer's evaluation exactly.

Example (local, MPS):
    CLIP_RETRIEVAL_ARTIFACT_ROOT=/Volumes/T7/Research/figures \
    ./.venv-eval/bin/python scripts/eval/compare_retrieval_pairs.py \
        --baseline-ckpt  /Volumes/T7/.../B0/B0_coco_s42/best_model.pth \
        --intervention-ckpt /Volumes/T7/.../B2/B2_coco_s123/best_model.pth \
        --baseline-label Base-min --intervention-label HN-Syntactic \
        --config configs/config_coco.yaml \
        --data-root /Volumes/T7/Research/experiments/datasets \
        --output-dir /Volumes/T7/Research/figures/retrieval_examples/basemin_vs_hnsyntactic \
        --top-k 6 --max-cases 40
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import create_image_text_dataloader  # noqa: E402
from src.setup import setup_config, setup_seed  # noqa: E402
from src.utils import chunked_matmul  # noqa: E402

# Reuse the exact ranking machinery from the failure-analysis tool.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
from extract_failures import (  # noqa: E402
    compute_gt_ranks,
    extract_embeddings,
    load_model,
)

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _apply_data_overrides(config: dict, data_root: str | None) -> dict:
    """Repoint dataset paths at an external data root, and force single-process loading.

    On the server the datasets are already mounted at the config's default paths, so
    data_root is omitted and only num_workers is forced to 0. data_root is used for local
    runs where images / Karpathy JSON live under an external root.
    """
    data = config["data"]
    # 0 workers: the dataloader's worker_init_fn is a local closure that can't be pickled
    # under macOS 'spawn'; single-process loading sidesteps it (fine for one eval pass).
    data["num_workers"] = 0
    if not data_root:
        return config

    root = Path(data_root)
    dataset = data.get("dataset", "coco")
    sub = "coco" if dataset == "coco" else "flickr30k"
    base = root / sub
    # images_path: coco config points at the dataset dir; flickr at the images subdir.
    data["images_path"] = str(base if dataset == "coco" else base / "flickr30k_images")
    json_name = "dataset_coco.json" if dataset == "coco" else "dataset_flickr30k.json"
    data["captions_path"] = str(base / "caption_datasets" / json_name)
    for key in ("seg_map_dir", "sam_feature_dir"):
        if data.get(key):
            data[key] = str(base / Path(data[key]).name)
    paths = config.get("paraphraser", {}).get("paths")
    if isinstance(paths, dict):
        config["paraphraser"]["paths"] = {k: str(base / Path(v).name) for k, v in paths.items()}
    elif isinstance(paths, list):
        config["paraphraser"]["paths"] = [str(base / Path(p).name) for p in paths]
    return config


def _ranks_for_checkpoint(ckpt_path, config, tokenizer, device):
    model = load_model(ckpt_path, config, device)
    loader = create_image_text_dataloader(config, tokenizer, split="test")
    dataset = loader.dataset
    img_e, txt_e, image_ids, unique_image_ids, first_occ = extract_embeddings(model, loader, device)
    sims = chunked_matmul(img_e, txt_e)  # [N_imgs, N_txts]
    i2t_ranks, t2i_ranks = compute_gt_ranks(sims, image_ids, unique_image_ids)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "sims": sims,
        "i2t": dict(i2t_ranks),  # img_idx -> rank
        "t2i": dict(t2i_ranks),  # cap_idx -> rank
        "samples": dataset.samples,
        "unique_image_ids": unique_image_ids.tolist(),
        "image_ids": image_ids.tolist(),
        "first_occ": first_occ,
    }


def _flipped_cases(base, interv, direction, broken_threshold, max_cases):
    """Return fixed and broken cases for one direction.

    fixed:  baseline rank > broken_threshold AND intervention rank == 0
    broken: baseline rank == 0 AND intervention rank > broken_threshold
    """
    fixed, broken = [], []
    keys = base[direction].keys() & interv[direction].keys()
    for idx in keys:
        rb = base[direction][idx]
        ri = interv[direction][idx]
        if rb > broken_threshold and ri == 0:
            fixed.append((idx, rb, ri, rb - ri))
        elif rb == 0 and ri > broken_threshold:
            broken.append((idx, rb, ri, ri - rb))
    fixed.sort(key=lambda x: x[3], reverse=True)
    broken.sort(key=lambda x: x[3], reverse=True)
    return fixed[:max_cases], broken[:max_cases]


def img_data_uri(images_root, sample, max_size=320):
    fp = sample.get("filepath") or ""
    path = fp if fp and os.path.isabs(fp) else os.path.join(images_root, fp or "", sample["filename"])
    if not os.path.isfile(path):
        path = os.path.join(images_root, sample["filename"])
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _topk_images(model_state, cap_idx, k, images_root):
    sims_t2i = model_state["sims"].t()
    top = sims_t2i[cap_idx].argsort(descending=True)[:k].tolist()
    out = []
    for img_idx in top:
        s = model_state["samples"][model_state["first_occ"][img_idx]]
        out.append({"image_id": model_state["unique_image_ids"][img_idx], "uri": img_data_uri(images_root, s)})
    return out


def _topk_captions(model_state, img_idx, k):
    top = model_state["sims"][img_idx].argsort(descending=True)[:k].tolist()
    return [model_state["samples"][c]["caption"] for c in top]


def build_case_records(base, interv, images_root, top_k, broken_threshold, max_cases):
    records = {"i2t": {"fixed": [], "broken": []}, "t2i": {"fixed": [], "broken": []}}
    for direction in ("i2t", "t2i"):
        fixed, broken = _flipped_cases(base, interv, direction, broken_threshold, max_cases)
        for label, cases in (("fixed", fixed), ("broken", broken)):
            for idx, rb, ri, _gap in cases:
                if direction == "t2i":
                    sample = base["samples"][idx]
                    gt = base["samples"][base["first_occ"][[i for i, iid in enumerate(base["unique_image_ids"]) if iid == sample["image_id"]][0]]] if sample["image_id"] in base["unique_image_ids"] else sample
                    rec = {
                        "query_type": "caption",
                        "query": sample["caption"],
                        "gt_image": img_data_uri(images_root, sample),
                        "baseline_rank": rb,
                        "intervention_rank": ri,
                        "baseline_topk": _topk_images(base, idx, top_k, images_root),
                        "intervention_topk": _topk_images(interv, idx, top_k, images_root),
                    }
                else:
                    dataset_idx = base["first_occ"][idx]
                    sample = base["samples"][dataset_idx]
                    rec = {
                        "query_type": "image",
                        "query_image": img_data_uri(images_root, sample),
                        "gt_caption": sample["caption"],
                        "baseline_rank": rb,
                        "intervention_rank": ri,
                        "baseline_topk": _topk_captions(base, idx, top_k),
                        "intervention_topk": _topk_captions(interv, idx, top_k),
                    }
                records[direction][label].append(rec)
    return records


def _render_topk(items, query_type):
    if query_type == "caption":  # retrieved images
        cells = "".join(f'<div class="cell"><img src="{it["uri"]}"/></div>' for it in items)
        return f'<div class="row">{cells}</div>'
    return "<ol class='caps'>" + "".join(f"<li>{c}</li>" for c in items) + "</ol>"


def build_html(records, baseline_label, intervention_label, top_k):
    blocks = []
    for direction, dlabel in (("t2i", "Text &rarr; Image"), ("i2t", "Image &rarr; Text")):
        for kind, klabel, color in (("fixed", "Fixed (baseline wrong &rarr; intervention right)", "#1b7837"),
                                    ("broken", "Broken (baseline right &rarr; intervention wrong)", "#b2182b")):
            cases = records[direction][kind]
            blocks.append(f'<h2 style="color:{color}">{dlabel} &mdash; {klabel} ({len(cases)})</h2>')
            for rec in cases:
                if rec["query_type"] == "caption":
                    query_html = f'<div class="query"><b>Query caption:</b> &ldquo;{rec["query"]}&rdquo;<br><b>GT image:</b><br><img src="{rec["gt_image"]}" class="gt"/></div>'
                else:
                    query_html = f'<div class="query"><b>Query image:</b><br><img src="{rec["query_image"]}" class="gt"/><br><b>GT caption:</b> &ldquo;{rec["gt_caption"]}&rdquo;</div>'
                blocks.append(f"""
                <div class="case">
                  {query_html}
                  <div class="models">
                    <div class="model"><div class="mlabel">{baseline_label} (rank {rec['baseline_rank']})</div>{_render_topk(rec['baseline_topk'], rec['query_type'])}</div>
                    <div class="model"><div class="mlabel">{intervention_label} (rank {rec['intervention_rank']})</div>{_render_topk(rec['intervention_topk'], rec['query_type'])}</div>
                  </div>
                </div>""")
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{baseline_label} vs {intervention_label}</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.case{{display:flex;gap:18px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #eee}}
.query{{flex:0 0 260px;font-size:12px}} .query .gt{{max-width:240px;border:2px solid #444;border-radius:4px}}
.models{{display:flex;gap:24px;flex:1}} .model{{flex:1}} .mlabel{{font-size:12px;font-weight:600;margin-bottom:4px}}
.row{{display:flex;gap:6px;flex-wrap:wrap}} .cell img{{height:84px;border-radius:3px}}
.caps{{font-size:12px;margin:0;padding-left:18px}} .caps li{{margin-bottom:2px}}
</style></head><body>
<h1>Retrieval flips: {baseline_label} vs {intervention_label}</h1>
<p style="font-size:12px;color:#666">Top-{top_k} retrieved shown per model. "rank" = ground-truth rank (0 = top-1, correct).</p>
{body}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--intervention-ckpt", required=True)
    ap.add_argument("--baseline-label", default="Baseline")
    ap.add_argument("--intervention-label", default="Intervention")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None,
                    help="External dataset root (local runs). Omit on the server where datasets "
                         "are mounted at the config's default paths.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-k", type=int, default=6, help="Retrieved items shown per model")
    ap.add_argument("--broken-threshold", type=int, default=4,
                    help="A query is 'wrong' for the flip test when its GT rank exceeds this")
    ap.add_argument("--max-cases", type=int, default=40, help="Max cases per (direction, kind)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    device = _device()
    logger.info(f"Device: {device}")

    config = setup_config(config_path=args.config, overrides=[])
    config = _apply_data_overrides(config, args.data_root)
    setup_seed(config["training"]["seed"])
    tokenizer = CLIPTokenizer.from_pretrained(config["model"]["image_model_name"])

    logger.info(f"Encoding baseline: {args.baseline_label}")
    base = _ranks_for_checkpoint(args.baseline_ckpt, config, tokenizer, device)
    logger.info(f"Encoding intervention: {args.intervention_label}")
    interv = _ranks_for_checkpoint(args.intervention_ckpt, config, tokenizer, device)

    images_root = config["data"]["images_path"]
    records = build_case_records(base, interv, images_root, args.top_k, args.broken_threshold, args.max_cases)

    for d in ("i2t", "t2i"):
        logger.info(f"{d}: fixed={len(records[d]['fixed'])} broken={len(records[d]['broken'])}")

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {d: {k: len(records[d][k]) for k in ("fixed", "broken")} for d in ("i2t", "t2i")}
    with open(os.path.join(args.output_dir, "flips.json"), "w") as f:
        json.dump({"summary": summary, "baseline": args.baseline_label,
                   "intervention": args.intervention_label, "cases": records}, f, indent=2)
    html = build_html(records, args.baseline_label, args.intervention_label, args.top_k)
    html_path = os.path.join(args.output_dir, "flips.html")
    with open(html_path, "w") as f:
        f.write(html)
    logger.info(f"Wrote {html_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
