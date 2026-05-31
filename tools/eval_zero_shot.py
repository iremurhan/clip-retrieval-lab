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
    Pretrained weights from HuggingFace only. Metrics match the base test
    evaluation surface: Recall@1/5/10, mAP@R, R-Precision, and COCO ECCV/CxC
    metrics when the COCO test split is evaluated.

Usage:
    python tools/eval_zero_shot.py --config configs/config_flickr30k.yaml --input rgb
    python tools/eval_zero_shot.py --config configs/config_flickr30k.yaml --input mask
    python tools/eval_zero_shot.py --config configs/config_coco.yaml --input rgb
    python tools/eval_zero_shot.py --config configs/config_coco.yaml --input mask

Outputs:
    Prints a formatted metric table, optionally dumps JSON, and can report the
    metrics to W&B under the zero_shot/<dataset>/<input>/ summary namespace.
    SugarCrepe and MMVP-VLM are enabled by default so the zero-shot baseline
    has the same post-training benchmark surface as trained runs.
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
from src.data import build_eval_transform
from src.eval.mmvp_vlm import evaluate_mmvp_vlm
from src.eval.sugarcrepe import evaluate_sugarcrepe
from src.metrics import (
    _build_gt_mappings,
    build_ranked_dicts,
    compute_eccv_metrics,
    compute_mapr_rprecision,
    compute_recall_at_k,
)
from src.setup import setup_config, setup_seed
from src.utils import chunked_matmul

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
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for deterministic library setup/bookkeeping. Zero-shot test eval is deterministic.")
    p.add_argument("--log-wandb", action="store_true",
                   help="Log metrics to a new or resumed W&B run.")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project. Default: config logging.wandb_project.")
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="Optional W&B run id to resume.")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Optional W&B run name. Default: zero_shot_clip_<dataset>_<input>.")
    p.add_argument("--wandb_group", type=str, default="zero_shot_clip",
                   help="W&B group for new zero-shot runs.")
    p.add_argument("--eval-sugarcrepe", action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate SugarCrepe and log zero_shot/<dataset>/<input>/sugarcrepe/* metrics.")
    p.add_argument("--sugarcrepe-data-dir", type=str, default="datasets/sugarcrepe",
                   help="Path to SugarCrepe JSON files.")
    p.add_argument("--sugarcrepe-images-dir", type=str, default="datasets/coco/val2017",
                   help="Path to COCO val2017 images used by SugarCrepe.")
    p.add_argument("--sugarcrepe-max-items-per-category", type=int, default=None,
                   help="Optional cap for SugarCrepe smoke tests.")
    p.add_argument("--eval-mmvp", action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate MMVP-VLM and log zero_shot/<dataset>/<input>/mmvp_vlm/* metrics.")
    p.add_argument("--mmvp-data-dir", type=str, default="datasets/mmvp_vlm",
                   help="Path to MMVP-VLM dataset directory.")
    return p.parse_args()


class CLIPZeroShotAdapter(torch.nn.Module):
    """Small adapter exposing the DualEncoder eval API for raw HuggingFace CLIP."""

    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        self.clip_model = clip_model

    def encode_image(self, images):
        feats = self.clip_model.get_image_features(pixel_values=images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        feats = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(feats, dim=-1)


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
    sentids = []

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
        sid = batch['sentid']
        sentids.extend(sid.tolist() if torch.is_tensor(sid) else sid)

    txt_embeds = torch.cat(embeddings, dim=0)  # [N_captions, D]
    image_ids = torch.tensor(image_ids, dtype=torch.long)  # [N_captions]
    sentids = torch.tensor(sentids, dtype=torch.long)  # [N_captions]
    return txt_embeds, image_ids, sentids


def compute_standard_metrics(
    img_embeds,
    txt_embeds,
    image_ids,
    unique_image_ids,
    sims,
) -> OrderedDict:
    r_t2i, r_i2t = compute_recall_at_k(
        img_embeds=img_embeds,
        txt_embeds=txt_embeds,
        image_ids=image_ids,
        unique_image_ids=unique_image_ids,
        k_values=[1, 5, 10],
        sims=sims,
    )

    sims_np = sims.t().numpy()  # [N_captions, N_images]
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

    return OrderedDict([
        ("r1_i2t", r_i2t[1]),
        ("r5_i2t", r_i2t[5]),
        ("r10_i2t", r_i2t[10]),
        ("r1_t2i", r_t2i[1]),
        ("r5_t2i", r_t2i[5]),
        ("r10_t2i", r_t2i[10]),
        ("mapr_i2t", mapr_rprec["mapr_i2t"]),
        ("mapr_t2i", mapr_rprec["mapr_t2i"]),
        ("rprecision_i2t", mapr_rprec["rprecision_i2t"]),
        ("rprecision_t2i", mapr_rprec["rprecision_t2i"]),
    ])


def add_coco_eccv_metrics(metrics: OrderedDict, sims, sentids, unique_image_ids) -> None:
    eccv_scores = compute_eccv_metrics(
        sims.t().numpy(),
        image_ids=unique_image_ids.tolist(),
        caption_ids=sentids.tolist(),
        dataset="coco",
    )
    if not eccv_scores:
        logger.warning("ECCV/CxC metrics unavailable; standard COCO metrics are still present.")
        return

    metrics.update(OrderedDict([
        ("coco_5k_r1_i2t", eccv_scores.get("coco_5k_r1", {}).get("i2t", 0)),
        ("coco_5k_r1_t2i", eccv_scores.get("coco_5k_r1", {}).get("t2i", 0)),
        ("coco_5k_r5_i2t", eccv_scores.get("coco_5k_r5", {}).get("i2t", 0)),
        ("coco_5k_r5_t2i", eccv_scores.get("coco_5k_r5", {}).get("t2i", 0)),
        ("coco_5k_r10_i2t", eccv_scores.get("coco_5k_r10", {}).get("i2t", 0)),
        ("coco_5k_r10_t2i", eccv_scores.get("coco_5k_r10", {}).get("t2i", 0)),
        ("coco_1k_r1_i2t", eccv_scores.get("coco_1k_r1", {}).get("i2t", 0)),
        ("coco_1k_r1_t2i", eccv_scores.get("coco_1k_r1", {}).get("t2i", 0)),
        ("coco_1k_r5_i2t", eccv_scores.get("coco_1k_r5", {}).get("i2t", 0)),
        ("coco_1k_r5_t2i", eccv_scores.get("coco_1k_r5", {}).get("t2i", 0)),
        ("coco_1k_r10_i2t", eccv_scores.get("coco_1k_r10", {}).get("i2t", 0)),
        ("coco_1k_r10_t2i", eccv_scores.get("coco_1k_r10", {}).get("t2i", 0)),
        ("eccv_map_at_r_i2t", eccv_scores.get("eccv_map_at_r", {}).get("i2t", 0)),
        ("eccv_map_at_r_t2i", eccv_scores.get("eccv_map_at_r", {}).get("t2i", 0)),
        ("eccv_rprecision_i2t", eccv_scores.get("eccv_rprecision", {}).get("i2t", 0)),
        ("eccv_rprecision_t2i", eccv_scores.get("eccv_rprecision", {}).get("t2i", 0)),
        ("cxc_r1_i2t", eccv_scores.get("cxc_r1", {}).get("i2t", 0)),
        ("cxc_r1_t2i", eccv_scores.get("cxc_r1", {}).get("t2i", 0)),
        ("cxc_r5_i2t", eccv_scores.get("cxc_r5", {}).get("i2t", 0)),
        ("cxc_r5_t2i", eccv_scores.get("cxc_r5", {}).get("t2i", 0)),
        ("cxc_r10_i2t", eccv_scores.get("cxc_r10", {}).get("i2t", 0)),
        ("cxc_r10_t2i", eccv_scores.get("cxc_r10", {}).get("t2i", 0)),
    ]))


def format_table(dataset: str, input_mode: str, metrics: dict) -> str:
    header = f"=== Zero-shot CLIP — dataset={dataset} | input={input_mode} ==="
    lines = [header]
    lines.append(f"{'Metric':<26} | {'Value':>8}")
    lines.append("-" * 38)
    for key, value in metrics.items():
        lines.append(f"{key:<26} | {value:>8.2f}")
    return "\n".join(lines)


def run_auxiliary_evals(args, config: dict, model_adapter, tokenizer, device: torch.device) -> OrderedDict:
    results = OrderedDict()
    transform = build_eval_transform(config["data"]["image_size"])
    max_length = config["data"].get("max_length", 77)

    if args.eval_sugarcrepe:
        logger.info("Running SugarCrepe zero-shot evaluation ...")
        sugar = evaluate_sugarcrepe(
            model=model_adapter,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            data_dir=args.sugarcrepe_data_dir,
            images_dir=args.sugarcrepe_images_dir,
            max_length=max_length,
            splits=("replace", "swap", "add"),
            max_items_per_category=args.sugarcrepe_max_items_per_category,
        )
        results["sugarcrepe"] = OrderedDict((k, float(v)) for k, v in sugar.items())

    if args.eval_mmvp:
        logger.info("Running MMVP-VLM zero-shot evaluation ...")
        mmvp, _ = evaluate_mmvp_vlm(
            model=model_adapter,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            data_dir=args.mmvp_data_dir,
            max_length=max_length,
        )
        results["mmvp_vlm"] = OrderedDict((k, float(v)) for k, v in mmvp.items())

    return results


def log_to_wandb(args, config: dict, payload: dict) -> None:
    import wandb

    dataset = payload["dataset"]
    input_mode = payload["input"]
    project = args.wandb_project or config.get("logging", {}).get("wandb_project", "clip-retrieval")
    run_name = args.wandb_run_name or f"zero_shot_clip_{dataset}_{input_mode}"
    init_kwargs = {
        "project": project,
        "name": run_name,
        "group": args.wandb_group,
        "config": {
            "run_id": "zero_shot_clip",
            "dataset": dataset,
            "input": input_mode,
            "model": payload["model"],
            "split": payload["split"],
            "seed": payload["seed"],
            "zero_shot": True,
            "use_seg_as_image": payload["use_seg_as_image"],
            "seed_affects_eval": False,
        },
        "tags": [
            "run_id:zero_shot_clip",
            f"dataset:{dataset}",
            f"input:{input_mode}",
            "zero_shot",
            "seed_irrelevant",
        ],
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "must"
    else:
        init_kwargs["resume"] = "allow"

    wandb.init(**init_kwargs)
    prefix = f"zero_shot/{dataset}/{input_mode}"
    log_data = {f"{prefix}/{key}": value for key, value in payload["metrics"].items()}
    for section in ("sugarcrepe", "mmvp_vlm"):
        for key, value in payload.get(section, {}).items():
            log_data[f"{prefix}/{section}/{key}"] = value
    wandb.log(log_data)
    for key, value in log_data.items():
        wandb.run.summary[key] = value
    wandb.finish()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

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
    model_adapter = CLIPZeroShotAdapter(model)
    model_adapter.eval()

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

    # 7. Retrieval metrics on a single similarity matrix.
    logger.info("Encoding captions ...")
    txt_embeds, image_ids, sentids = encode_texts(model, test_loader, device)
    logger.info(f"  -> {txt_embeds.shape[0]} captions")

    logger.info("Computing base metric suite ...")
    sims = chunked_matmul(img_embeds, txt_embeds)
    metrics = compute_standard_metrics(img_embeds, txt_embeds, image_ids, unique_image_ids, sims)

    dataset = config['data']['dataset']
    if dataset == "coco":
        add_coco_eccv_metrics(metrics, sims, sentids, unique_image_ids)

    print("\n" + format_table(dataset, args.input, metrics) + "\n")
    auxiliary_results = run_auxiliary_evals(args, config, model_adapter, tokenizer, device)

    payload = OrderedDict([
        ("dataset", dataset),
        ("input", args.input),
        ("model", model_name),
        ("split", "test"),
        ("seed", args.seed),
        ("seed_affects_eval", False),
        ("n_unique_images", int(img_embeds.shape[0])),
        ("n_captions", int(txt_embeds.shape[0])),
        ("metrics", metrics),
        *auxiliary_results.items(),
        ("palette_size", config.get('data', {}).get('seg_palette_size')),
        ("seg_map_dir", config.get('data', {}).get('seg_map_dir')),
        ("use_seg_as_image", config.get('data', {}).get('use_seg_as_image', False)),
    ])

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote results to {args.output}")

    if args.log_wandb:
        log_to_wandb(args, config, payload)


if __name__ == "__main__":
    main()
