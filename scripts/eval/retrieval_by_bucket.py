"""
Compute retrieval R@K by caption-complexity bucket on standard test sets.

Checkpoint discovery is intentionally driven by a fresh wandb.Api() query on
every run. Local checkpoint paths are only used after being cross-checked
against those live WandB runs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.discover_pending_sugarcrepe import (  # noqa: E402
    parse_wandb_run_id_from_log,
    parse_wandb_run_name_from_log,
)
from scripts.eval.eval_cross_dataset import (  # noqa: E402
    build_eval_config,
    build_tokenizer,
    dataset_paths,
    select_device,
)
from scripts.eval.eval_sugarcrepe import load_model_from_checkpoint  # noqa: E402
from src.data import create_image_text_dataloader  # noqa: E402
from src.metrics import _build_gt_mappings  # noqa: E402
from src.utils import chunked_matmul  # noqa: E402


logger = logging.getLogger(__name__)

PROJECT = "iremurhan-bogazici-university/clip-retrieval"
DATASETS = ("coco", "flickr30k")
EXCLUDE = {"B0v2", "B0plus_fixed", "B5_seg", "B0_proj1024"}
K_VALUES = (1, 5, 10)
DEFAULT_RESULTS_ROOT = Path("/Volumes/T7/Research/experiments/results")
DEFAULT_DATA_ROOT = Path("/Volumes/T7/Research/experiments/datasets")
ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
FIG_ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_FIG_ARTIFACT_ROOT", ARTIFACT_ROOT / "figs")
)
DEFAULT_BUCKET_DIR = Path(os.environ.get("BUCKET_DIR", FIG_ARTIFACT_ROOT / "cache"))
DEFAULT_OUTPUT = Path(
    os.environ.get("RETRIEVAL_BY_BUCKET_OUTPUT", FIG_ARTIFACT_ROOT / "cache" / "retrieval_by_bucket.csv")
)
DEFAULT_EMBED_CACHE = Path(
    os.environ.get("RETRIEVAL_BY_BUCKET_EMBED_CACHE", ARTIFACT_ROOT / "cache" / "retrieval_by_bucket_embeddings")
)


def normalize_dataset(value: object) -> str:
    text = str(value or "").strip()
    if text == "flickr":
        return "flickr30k"
    return text


def nested_get(mapping: dict, *keys: str):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def run_config_value(run, key: str):
    config = run.config or {}
    if key in config:
        return config[key]
    slash_key = f"config/{key}"
    if slash_key in config:
        return config[slash_key]
    return nested_get(config, *key.split("/"))


def parse_identity_from_name(name: str) -> tuple[str | None, str | None, int | None]:
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)(?:_\d+)?$")
    match = pattern.match(name or "")
    if not match:
        return None, None, None
    return (
        match.group("run"),
        normalize_dataset(match.group("dataset")),
        int(match.group("seed")),
    )


def fresh_wandb_entries(project: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    entries: list[dict] = []
    for run in api.runs(project):
        parsed_run_id, parsed_dataset, parsed_seed = parse_identity_from_name(run.name)
        run_id = run_config_value(run, "run_id") or parsed_run_id
        dataset = normalize_dataset(run_config_value(run, "dataset") or parsed_dataset)
        seed = run_config_value(run, "seed")
        if seed is None:
            seed = parsed_seed
        if run_id is None or dataset not in DATASETS or seed is None:
            continue
        run_id = str(run_id)
        if run_id in EXCLUDE:
            continue
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            continue
        entries.append(
            {
                "run_id": run_id,
                "seed": seed,
                "dataset": dataset,
                "wandb_run_name": run.name,
                "wandb_run_id": run.id,
                "state": run.state,
            }
        )
    entries.sort(key=lambda row: (row["dataset"], row["run_id"], row["seed"], row["wandb_run_name"]))
    return entries


def discover_local_checkpoints(results_root: Path) -> dict[tuple[str, str, int], dict]:
    out: dict[tuple[str, str, int], dict] = {}
    for checkpoint in sorted(results_root.rglob("best_model.pth")):
        log_path = checkpoint.with_name("training.log")
        wandb_id = parse_wandb_run_id_from_log(str(log_path)) or ""
        wandb_name = parse_wandb_run_name_from_log(str(log_path)) or checkpoint.parent.name
        run_id, dataset, seed = parse_identity_from_name(wandb_name)
        if run_id is None or dataset is None or seed is None:
            run_id, dataset, seed = parse_identity_from_name(checkpoint.parent.name)
        if run_id is None or dataset is None or seed is None:
            continue
        out[(run_id, dataset, seed)] = {
            "checkpoint_path": str(checkpoint),
            "wandb_run_name": wandb_name,
            "wandb_run_id": wandb_id,
        }
    return out


def build_eval_entries(project: str, results_root: Path, strict: bool = False) -> list[dict]:
    wandb_entries = fresh_wandb_entries(project)
    checkpoints = discover_local_checkpoints(results_root)
    rows: list[dict] = []
    missing: list[dict] = []
    for entry in wandb_entries:
        key = (entry["run_id"], entry["dataset"], entry["seed"])
        local = checkpoints.get(key)
        if local is None:
            missing.append(entry)
            continue
        rows.append({**entry, **local})

    print(f"Fresh WandB entries after excludes: {len(wandb_entries)}")
    print(f"Matched local checkpoints: {len(rows)}")
    if missing:
        print("WARNING: WandB runs without matching local checkpoint:")
        for entry in missing[:50]:
            print(f"  {entry['wandb_run_name']} ({entry['wandb_run_id']})")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
        if strict:
            raise FileNotFoundError(f"{len(missing)} WandB runs lacked local checkpoints.")
    if not rows:
        raise RuntimeError("No WandB runs matched local checkpoints.")
    return rows


def embed_cache_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    safe = run_id.replace("/", "_")
    return cache_dir / f"{dataset}_{safe}_s{seed}.pt"


def maybe_cpu_fallback_for_mps(model, config: dict, device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    if device.type != "mps":
        return model, device
    try:
        image_size = int(config["data"]["image_size"])
        max_length = int(config["data"]["max_length"])
        dummy_img = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_ids = torch.zeros(1, max(5, min(max_length, 8)), dtype=torch.long, device=device)
        dummy_mask = torch.ones_like(dummy_ids)
        with torch.no_grad():
            model.encode_image(dummy_img)
            model.encode_text(dummy_ids, dummy_mask)
        logger.info("MPS smoke test passed.")
    except Exception as exc:
        logger.error("MPS smoke test failed (%s). Falling back to CPU.", exc)
        device = torch.device("cpu")
        model = model.to(device)
    return model, device


@torch.no_grad()
def extract_embeddings_unique_images(model, loader, device: torch.device, use_amp: bool) -> tuple:
    """Encode every caption but only the first occurrence of each test image."""
    model.eval()
    img_embeds_list = []
    txt_embeds_list = []
    image_ids_list = []
    unique_image_ids_list = []
    sentids_list = []
    seen_image_ids: set[int] = set()

    use_seg_ids = loader.dataset.seg_loader is not None
    use_sam_features = loader.dataset.sam_feature_loader is not None

    for idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        seg_ids = batch["seg_ids"].to(device, non_blocking=True) if use_seg_ids else None
        sam_features = batch["sam_features"].to(device, non_blocking=True) if use_sam_features else None

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            text_embeds = model.encode_text(input_ids, attention_mask)

        new_positions = []
        for pos, image_id in enumerate(batch["image_id"].tolist()):
            image_id = int(image_id)
            if image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            unique_image_ids_list.append(image_id)
            new_positions.append(pos)

        if new_positions:
            pos_tensor = torch.tensor(new_positions, dtype=torch.long, device=device)
            unique_images = images.index_select(0, pos_tensor)
            unique_seg_ids = seg_ids.index_select(0, pos_tensor) if seg_ids is not None else None
            unique_sam = sam_features.index_select(0, pos_tensor) if sam_features is not None else None
            image_chunks = []
            image_chunk_size = 4 if device.type == "mps" else len(new_positions)
            for start in range(0, unique_images.shape[0], image_chunk_size):
                end = start + image_chunk_size
                image_chunk = unique_images[start:end]
                seg_chunk = unique_seg_ids[start:end] if unique_seg_ids is not None else None
                sam_chunk = unique_sam[start:end] if unique_sam is not None else None
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    if sam_chunk is not None:
                        image_embeds = model.encode_image(image_chunk, seg_ids=seg_chunk, sam_features=sam_chunk)
                    elif seg_chunk is not None:
                        image_embeds = model.encode_image(image_chunk, seg_ids=seg_chunk)
                    else:
                        image_embeds = model.encode_image(image_chunk)
                image_chunks.append(image_embeds.float().cpu())
                del image_embeds
            img_embeds_list.append(torch.cat(image_chunks, dim=0))

        txt_embeds_list.append(text_embeds.float().cpu())
        image_ids_list.append(batch["image_id"].cpu())
        sentids_list.append(batch["sentid"].cpu())

        if idx % 20 == 0:
            logger.info("Encoded %d/%d batches", idx, len(loader))

    img_embeds_unique = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    image_ids = torch.cat(image_ids_list, dim=0)
    sentids = torch.cat(sentids_list, dim=0)
    unique_image_ids = torch.tensor(unique_image_ids_list, dtype=image_ids.dtype)
    return img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, unique_image_ids_list


def load_or_compute_embeddings(
    entry: dict,
    data_root: Path,
    cache_dir: Path,
    device_arg: str,
    batch_size: int | None,
    num_workers: int,
    force_embeddings: bool,
) -> dict:
    path = embed_cache_path(cache_dir, entry["run_id"], entry["dataset"], entry["seed"])
    if path.exists() and not force_embeddings:
        logger.info("Loading embedding cache: %s", path)
        return torch.load(path, map_location="cpu", weights_only=False)

    device = select_device(device_arg)
    model, checkpoint_config = load_model_from_checkpoint(entry["checkpoint_path"], device)
    eval_config = build_eval_config(checkpoint_config, entry["dataset"], data_root, batch_size, num_workers)
    # Ensure standard in-domain test paths even if checkpoint config used older relative paths.
    eval_config.setdefault("data", {}).update(dataset_paths(entry["dataset"], data_root))
    tokenizer = build_tokenizer(checkpoint_config)
    model, device = maybe_cpu_fallback_for_mps(model, eval_config, device)
    loader = create_image_text_dataloader(eval_config, tokenizer, split="test")
    logger.info(
        "Encoding %s %s seed=%s on %s: %d captions",
        entry["run_id"],
        entry["dataset"],
        entry["seed"],
        device,
        len(loader.dataset),
    )
    embeddings = extract_embeddings_unique_images(model, loader, device, use_amp=(device.type == "cuda"))
    img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, unique_image_ids_list = embeddings
    payload = {
        "run_id": entry["run_id"],
        "seed": int(entry["seed"]),
        "dataset": entry["dataset"],
        "checkpoint_path": entry["checkpoint_path"],
        "image_embeds": F.normalize(img_embeds_unique.float(), p=2, dim=1),
        "text_embeds": F.normalize(txt_embeds.float(), p=2, dim=1),
        "image_ids": image_ids.cpu(),
        "unique_image_ids": unique_image_ids.cpu(),
        "sentids": sentids.cpu(),
        "unique_image_ids_list": list(unique_image_ids_list),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Wrote embedding cache: %s", path)
    del model, loader
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    gc.collect()
    return payload


def load_buckets(dataset: str, bucket_dir: Path) -> pd.DataFrame:
    path = bucket_dir / f"caption_buckets_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Caption bucket CSV missing: {path}. Run build_caption_buckets.py first.")
    df = pd.read_csv(path)
    required = {"caption_id", "image_id", "token_quartile", "concept_quartile"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df


def recall_from_ranks(ranks: torch.Tensor, k: int) -> float:
    if ranks.numel() == 0:
        return float("nan")
    return float((ranks < k).float().mean().item() * 100.0)


def recall_from_hits(hits: torch.Tensor) -> float:
    if hits.numel() == 0:
        return float("nan")
    return float(hits.float().mean().item() * 100.0)


def compute_t2i_rows(payload: dict, bucket_df: pd.DataFrame) -> list[dict]:
    image_embeds = payload["image_embeds"]
    text_embeds = payload["text_embeds"]
    image_ids = payload["image_ids"]
    unique_image_ids = payload["unique_image_ids"]
    sentids = payload["sentids"]

    sims = chunked_matmul(image_embeds, text_embeds)  # [images, captions]
    _, caption_to_image_idx, _ = _build_gt_mappings(image_ids, unique_image_ids)
    max_k = max(K_VALUES)
    topk = torch.topk(sims.t(), k=max_k, dim=1).indices  # [captions, max_k]
    correct = topk.eq(caption_to_image_idx.unsqueeze(1))

    bucket = bucket_df.set_index("caption_id")
    meta = pd.DataFrame({"caption_idx": range(len(sentids)), "caption_id": sentids.tolist()})
    meta = meta.join(bucket[["token_quartile", "concept_quartile"]], on="caption_id", how="left")
    if meta[["token_quartile", "concept_quartile"]].isna().any().any():
        missing = int(meta[["token_quartile", "concept_quartile"]].isna().any(axis=1).sum())
        raise ValueError(f"{missing} encoded captions are missing bucket annotations.")

    rows: list[dict] = []
    selections = [("all", "All", meta.index.to_numpy())]
    for dim in ["token_quartile", "concept_quartile"]:
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            selections.append((dim, q, meta.index[meta[dim].eq(q)].to_numpy()))

    for bucket_dim, bucket_value, indices_np in selections:
        indices = torch.tensor(indices_np, dtype=torch.long)
        bucket_correct = correct.index_select(0, indices)
        for k in K_VALUES:
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "seed": payload["seed"],
                    "dataset": payload["dataset"],
                    "direction": "t2i",
                    "bucket_dim": bucket_dim,
                    "bucket_value": bucket_value,
                    "k": k,
                    "recall": recall_from_hits(bucket_correct[:, :k].any(dim=1)),
                    "n_queries": int(bucket_correct.shape[0]),
                }
            )
    return rows


def compute_i2t_all_rows(payload: dict) -> list[dict]:
    image_embeds = payload["image_embeds"]
    text_embeds = payload["text_embeds"]
    image_ids = payload["image_ids"]
    unique_image_ids = payload["unique_image_ids"]
    sims = chunked_matmul(image_embeds, text_embeds)
    _, _, image_to_caption_indices = _build_gt_mappings(image_ids, unique_image_ids)
    max_k = max(K_VALUES)
    topk = torch.topk(sims, k=max_k, dim=1).indices
    rows: list[dict] = []
    hits_by_k: dict[int, list[bool]] = {k: [] for k in K_VALUES}
    for image_idx in range(sims.shape[0]):
        gt = torch.tensor(sorted(image_to_caption_indices[image_idx]), dtype=torch.long)
        correct = torch.isin(topk[image_idx], gt)
        for k in K_VALUES:
            hits_by_k[k].append(bool(correct[:k].any().item()))
    for k in K_VALUES:
        hits = torch.tensor(hits_by_k[k], dtype=torch.bool)
        rows.append(
            {
                "run_id": payload["run_id"],
                "seed": payload["seed"],
                "dataset": payload["dataset"],
                "direction": "i2t",
                "bucket_dim": "all",
                "bucket_value": "All",
                "k": k,
                "recall": recall_from_hits(hits),
                "n_queries": int(hits.numel()),
            }
        )
    return rows


def write_manifest_snapshot(entries: list[dict], output_path: Path) -> None:
    path = output_path.with_name("retrieval_by_bucket_wandb_manifest_snapshot.csv")
    fieldnames = ["run_id", "seed", "dataset", "wandb_run_name", "wandb_run_id", "state", "checkpoint_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    print(f"Wrote fresh WandB manifest snapshot: {path}")


def print_signal_report(df: pd.DataFrame) -> None:
    r1 = df[(df["direction"] == "t2i") & (df["k"] == 1) & (df["bucket_dim"].isin(["token_quartile", "concept_quartile"]))]
    agg = r1.groupby(["run_id", "dataset", "bucket_dim", "bucket_value"], as_index=False)["recall"].mean()
    pivot = agg.pivot_table(index=["run_id", "dataset", "bucket_dim"], columns="bucket_value", values="recall")
    flagged = []
    for idx, row in pivot.iterrows():
        if "Q1" not in row or "Q4" not in row or pd.isna(row["Q1"]) or pd.isna(row["Q4"]):
            continue
        delta = float(row["Q4"] - row["Q1"])
        if abs(delta) > 2.0:
            flagged.append((*idx, delta))
    if flagged:
        print("Configs with >2pp R@1 Q1-Q4 bucket difference:")
        for run_id, dataset, bucket_dim, delta in flagged:
            print(f"  {run_id} {dataset} {bucket_dim}: Q4-Q1={delta:+.2f} pp")
    else:
        print("No bucket-conditional signal detected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute R@K by caption-complexity bucket.")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bucket-dir", type=Path, default=DEFAULT_BUCKET_DIR)
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_EMBED_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--strict-wandb", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit after fresh WandB matching.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch WandB/checkpoint manifest and exit before encoding.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = build_eval_entries(args.project, args.results_root, strict=args.strict_wandb)
    if args.limit is not None:
        entries = entries[: args.limit]
        print(f"Debug limit applied: evaluating {len(entries)} entries")
    write_manifest_snapshot(entries, args.output)
    if args.dry_run:
        for entry in entries:
            print(f"DRY RUN {entry['run_id']} {entry['dataset']} seed={entry['seed']} -> {entry['checkpoint_path']}")
        return
    if args.output.exists() and not args.force_output:
        print(f"Output exists, leaving unchanged after fresh WandB check: {args.output}")
        existing = pd.read_csv(args.output)
        print_signal_report(existing)
        return

    bucket_by_dataset = {dataset: load_buckets(dataset, args.bucket_dir) for dataset in DATASETS}
    rows: list[dict] = []
    for idx, entry in enumerate(entries, start=1):
        print(f"\n[{idx}/{len(entries)}] {entry['run_id']} {entry['dataset']} seed={entry['seed']}")
        payload = load_or_compute_embeddings(
            entry,
            args.data_root,
            args.embedding_cache_dir,
            args.device,
            args.batch_size,
            args.num_workers,
            args.force_embeddings,
        )
        rows.extend(compute_t2i_rows(payload, bucket_by_dataset[entry["dataset"]]))
        rows.extend(compute_i2t_all_rows(payload))
        del payload
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        gc.collect()

    out = pd.DataFrame(
        rows,
        columns=["run_id", "seed", "dataset", "direction", "bucket_dim", "bucket_value", "k", "recall", "n_queries"],
    ).sort_values(["dataset", "run_id", "seed", "direction", "bucket_dim", "bucket_value", "k"])
    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")
    print_signal_report(out)


if __name__ == "__main__":
    main()
