from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.thesis_labels import LABEL_COLUMNS, load_thesis_label_map


DEFAULT_ENTITY = "iremurhan-bogazici-university"
DEFAULT_PROJECT = "clip-retrieval"
ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/figures")
)
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "clean"
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", "/Volumes/T7/Research/experiments/results")
)
CONFIG_BASE_PATH = Path(__file__).resolve().parents[2] / "configs" / "config_base.yaml"
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "registry.yaml"

DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}
IDENTITY_RE = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)(?:_\d+)?$")

SUGARCREPE_CATEGORIES = {
    "add_att",
    "add_obj",
    "replace_att",
    "replace_obj",
    "replace_rel",
    "swap_att",
    "swap_obj",
}
MMVP_PATTERNS = {
    "orientation",
    "presence",
    "state",
    "quantity",
    "spatial",
    "color",
    "structural",
    "text_rendering",
    "viewpoint",
}
OOD_RE = re.compile(r"^ood/(?P<src>coco|flickr30k|flickr)_to_(?P<dst>coco|flickr30k|flickr)/(?P<metric>.+)$")
NON_METRIC_KEYS = {
    "epoch",
    "global_step",
    "trainer/global_step",
    "train/epoch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export W&B runs to cleaned manifest and normalized metric tables."
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-samples", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit on number of W&B runs.")
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def as_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def json_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if str(key).startswith("_"):
            continue
        column = f"{prefix}/{key}"
        if isinstance(value, dict):
            out.update(flatten(column, value))
        else:
            out[column] = json_cell(value)
    return out


def normalize_dataset(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return DATASET_ALIASES.get(text, text or None)


def parse_identity_from_name(name: str | None) -> tuple[str | None, str | None, int | None]:
    match = IDENTITY_RE.match(name or "")
    if not match:
        return None, None, None
    return (
        match.group("run"),
        normalize_dataset(match.group("dataset")),
        int(match.group("seed")),
    )


def nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def config_value(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    return nested_get(config, *key.split("/"))


def first_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config_value(config, key)
        if not is_missing(value):
            return value
    return None


@lru_cache(maxsize=1)
def registry_unfreeze_layers() -> dict[str, Any]:
    with CONFIG_BASE_PATH.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        registry = (yaml.safe_load(f) or {}).get("runs", {})

    base_unfreeze = nested_get(base_config, "model", "unfreeze_vision_layers")
    resolved: dict[str, Any] = {}

    def resolve(run_id: str) -> Any:
        if run_id in resolved:
            return resolved[run_id]
        entry = registry.get(run_id) or {}
        parent = entry.get("parent")
        value = resolve(str(parent)) if parent else base_unfreeze
        overrides = entry.get("overrides") or {}
        if "model.unfreeze_vision_layers" in overrides:
            value = overrides["model.unfreeze_vision_layers"]
        elif isinstance(overrides.get("model"), dict) and "unfreeze_vision_layers" in overrides["model"]:
            value = overrides["model"]["unfreeze_vision_layers"]
        resolved[run_id] = value
        return value

    for run_id in registry:
        resolve(str(run_id))
    return resolved


def run_identity(run: Any) -> dict[str, Any]:
    config = dict(run.config or {})
    parsed_run_id, parsed_dataset, parsed_seed = parse_identity_from_name(getattr(run, "name", None))
    run_id = config_value(config, "run_id") or parsed_run_id
    dataset = normalize_dataset(config_value(config, "dataset") or parsed_dataset)
    seed = config_value(config, "seed")
    if seed is None:
        seed = parsed_seed
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    internal_run_id = str(run_id) if run_id is not None else None
    return {
        "internal_run_id": internal_run_id,
        "registry_id": internal_run_id,
        "dataset": dataset,
        "seed": seed,
    }


def parse_wandb_run_id_from_log(training_log_path: Path) -> str | None:
    if not training_log_path.is_file():
        return None
    patterns = [
        re.compile(r"wandb:\s+setting up run\s+(\S+)"),
        re.compile(r"run-\d{8}_\d{6}-(\w+)"),
    ]
    try:
        with training_log_path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx > 100:
                    break
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        return match.group(1)
    except OSError:
        return None
    return None


def parse_wandb_run_name_from_log(training_log_path: Path) -> str | None:
    if not training_log_path.is_file():
        return None
    pattern = re.compile(r"wandb:\s+Syncing run\s+(\S+)")
    try:
        with training_log_path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx > 100:
                    break
                match = pattern.search(line)
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def discover_checkpoints(results_root: Path) -> tuple[dict[str, str], dict[tuple[str, str, int], str]]:
    by_wandb_id: dict[str, str] = {}
    by_identity: dict[tuple[str, str, int], str] = {}
    if not results_root.exists():
        return by_wandb_id, by_identity

    for checkpoint in sorted(results_root.rglob("best_model.pth")):
        log_path = checkpoint.with_name("training.log")
        wandb_id = parse_wandb_run_id_from_log(log_path)
        wandb_name = parse_wandb_run_name_from_log(log_path) or checkpoint.parent.name
        run_id, dataset, seed = parse_identity_from_name(wandb_name)
        if run_id is None or dataset is None or seed is None:
            run_id, dataset, seed = parse_identity_from_name(checkpoint.parent.name)
        if wandb_id:
            by_wandb_id[wandb_id] = str(checkpoint)
        if run_id is not None and dataset is not None and seed is not None:
            by_identity[(run_id, dataset, seed)] = str(checkpoint)
    return by_wandb_id, by_identity


def normalize_metric_key(raw_key: str, dataset: str | None = None) -> dict[str, Any] | None:
    key = raw_key.removeprefix("summary/").removeprefix("history/")
    metric = key
    split = None
    benchmark = None
    direction = None
    k_value = None
    family = key.split("/", 1)[0] if "/" in key else "other"

    ood_match = OOD_RE.match(key)
    if ood_match:
        src = normalize_dataset(ood_match.group("src"))
        dst = normalize_dataset(ood_match.group("dst"))
        inner = normalize_metric_key(ood_match.group("metric"), dataset=dst)
        if inner is None:
            return None
        inner_metric = inner["metric"]
        standard_prefix = f"test/{dst}/"
        if inner_metric.startswith(standard_prefix):
            inner_metric = inner_metric[len(standard_prefix) :]
        elif inner_metric.startswith("test/"):
            inner_metric = inner_metric[len("test/") :]
        metric = f"ood/{src}_to_{dst}/{inner_metric}"
        return {
            **inner,
            "metric": metric,
            "family": "ood",
            "split": "ood",
            "benchmark": f"{src}_to_{dst}",
            "source_metric": key,
        }

    bare_standard_re = re.compile(r"^(?:r\d+|mapr|rprecision)_(?:i2t|t2i)$")
    if bare_standard_re.match(key):
        metric = f"test/{dataset or 'unknown'}/{key}"
        split = "test"
        benchmark = dataset
        family = "retrieval"
    elif key.startswith(("coco_5k_", "coco_1k_", "cxc_", "eccv_")):
        replacements = [
            ("coco_5k_", "test/coco_5k/"),
            ("coco_1k_", "test/coco_1k/"),
            ("cxc_", "test/cxc/"),
            ("eccv_", "test/eccv/"),
        ]
        for prefix, target in replacements:
            if key.startswith(prefix):
                metric = target + key[len(prefix) :]
                split = "test"
                benchmark = target.rstrip("/").split("/")[-1]
                family = "retrieval"
                break

    if key.startswith("test/"):
        rest = key[len("test/") :]
        if dataset == "flickr30k" and re.match(r"^(r|mapr|rprecision)", rest):
            metric = f"test/flickr30k/{rest}"
            split = "test"
            benchmark = "flickr30k"
        elif dataset == "coco" and re.match(r"^(r|mapr|rprecision)", rest):
            metric = f"test/coco_standard/{rest}"
            split = "test"
            benchmark = "coco_standard"
        else:
            replacements = [
                ("coco_5k_", "test/coco_5k/"),
                ("coco_1k_", "test/coco_1k/"),
                ("cxc_", "test/cxc/"),
                ("eccv_", "test/eccv/"),
            ]
            for prefix, target in replacements:
                if rest.startswith(prefix):
                    metric = target + rest[len(prefix) :]
                    split = "test"
                    benchmark = target.rstrip("/").split("/")[-1]
                    break
        family = "retrieval"

    elif key.startswith("val/"):
        rest = key[len("val/") :]
        metric = f"val/{dataset or 'unknown'}/{rest}"
        split = "val"
        benchmark = dataset
        family = "retrieval"

    elif key.startswith("train/"):
        rest = key[len("train/") :]
        aliases = {
            "loss_inter": "loss_clip",
            "loss_total": "loss_total",
            "loss_intra_img": "loss_intra_img",
            "loss_intra_txt": "loss_intra_txt",
        }
        metric = f"train/{aliases.get(rest, rest)}"
        split = "train"
        family = "training"

    elif key.startswith("sugarcrepe/"):
        rest = key[len("sugarcrepe/") :]
        subcat = "overall" if rest in {"macro_avg", "overall"} else rest
        metric = f"sugarcrepe/{subcat}/accuracy"
        benchmark = "sugarcrepe"
        family = "sugarcrepe"

    elif key.startswith("mmvp_vlm/"):
        rest = key[len("mmvp_vlm/") :]
        metric = f"mmvp_vlm/{rest}/accuracy"
        benchmark = "mmvp_vlm"
        family = "mmvp_vlm"

    recall_match = re.search(r"(?:^|/)(r)(?P<k>\d+)_(?P<direction>i2t|t2i)$", metric)
    if recall_match:
        direction = recall_match.group("direction")
        k_value = int(recall_match.group("k"))
    else:
        dir_match = re.search(r"_(?P<direction>i2t|t2i)$", metric)
        if dir_match:
            direction = dir_match.group("direction")

    return {
        "metric": metric,
        "family": family,
        "split": split,
        "benchmark": benchmark,
        "direction": direction,
        "k": k_value,
        "source_metric": key,
    }


def metric_rows_for_mapping(
    mapping: dict[str, Any],
    *,
    source: str,
    run_meta: dict[str, Any],
    step: int | None = None,
    epoch: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = run_meta.get("dataset")
    for raw_key, value in mapping.items():
        metric_key = str(raw_key).removeprefix("summary/").removeprefix("history/")
        if metric_key.startswith("_") or metric_key in NON_METRIC_KEYS:
            continue
        numeric = as_float(value)
        if numeric is None:
            continue
        normalized = normalize_metric_key(raw_key, dataset=dataset)
        if normalized is None:
            continue
        rows.append(
            {
                **run_meta,
                "source": source,
                "step": step,
                "epoch": epoch,
                "raw_metric": raw_key,
                "metric": normalized["metric"],
                "family": normalized["family"],
                "split": normalized["split"],
                "benchmark": normalized["benchmark"],
                "direction": normalized["direction"],
                "k": normalized["k"],
                "value": numeric,
            }
        )
    return rows


def history_rows(run: Any, run_meta: dict[str, Any], samples: int) -> list[dict[str, Any]]:
    import pandas as pd

    try:
        history = run.history(pandas=True, samples=samples)
        if not isinstance(history, pd.DataFrame) or history.empty:
            history = pd.DataFrame(list(run.scan_history(page_size=10_000)))
    except Exception as exc:
        print(f"WARNING: could not fetch history for {run.name} ({run.id}): {exc}")
        return []

    if history.empty:
        return []

    rows: list[dict[str, Any]] = []
    for record in history.to_dict(orient="records"):
        step = record.get("_step")
        epoch = record.get("epoch", record.get("train/epoch"))
        try:
            step = int(step) if not is_missing(step) else None
        except (TypeError, ValueError):
            step = None
        try:
            epoch = float(epoch) if not is_missing(epoch) else None
        except (TypeError, ValueError):
            epoch = None
        rows.extend(metric_rows_for_mapping(record, source="history", run_meta=run_meta, step=step, epoch=epoch))
    return rows


def completion_status(summary_metrics: set[str], dataset: str | None) -> tuple[list[str], list[str]]:
    expected: list[tuple[str, list[str]]] = [("sugarcrepe", ["sugarcrepe/overall/accuracy"])]
    expected.append(("mmvp_vlm", ["mmvp_vlm/overall/accuracy"]))

    if dataset == "coco":
        expected.extend(
            [
                ("flickr30k_retrieval_ood", ["ood/coco_to_flickr30k/r1_i2t", "ood/coco_to_flickr30k/r1_t2i"]),
                ("coco_5k_retrieval", ["test/coco_5k/r1_i2t", "test/coco_5k/r1_t2i"]),
                ("coco_1k_retrieval", ["test/coco_1k/r1_i2t", "test/coco_1k/r1_t2i"]),
                ("cxc", ["test/cxc/r1_i2t", "test/cxc/r1_t2i"]),
                ("eccv_captions", ["test/eccv/map_at_r_i2t", "test/eccv/map_at_r_t2i"]),
            ]
        )
    elif dataset == "flickr30k":
        expected.extend(
            [
                ("flickr30k_retrieval", ["test/flickr30k/r1_i2t", "test/flickr30k/r1_t2i"]),
                ("coco_retrieval_ood", ["ood/flickr30k_to_coco/r1_i2t", "ood/flickr30k_to_coco/r1_t2i"]),
                ("ood_cxc", ["ood/flickr30k_to_coco/cxc/r1_i2t", "ood/flickr30k_to_coco/cxc/r1_t2i"]),
                (
                    "ood_eccv_captions",
                    ["ood/flickr30k_to_coco/eccv/map_at_r_i2t", "ood/flickr30k_to_coco/eccv/map_at_r_t2i"],
                ),
            ]
        )

    completed = []
    missing = []
    for name, keys in expected:
        if any(key in summary_metrics for key in keys):
            completed.append(name)
        else:
            missing.append(name)
    return completed, missing


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_missing_csv(path: Path, manifest_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "wandb_run_id",
        "wandb_run_name",
        "internal_run_id",
        "canonical_run_id",
        "thesis_label",
        "display_label",
        "latex_label",
        "intervention_group",
        "reference_label",
        "label_kind",
        "include_in_main_results",
        "include_in_sweep_figures",
        "sweep_display_label",
        "sweep_index",
        "is_excluded",
        "exclude_reason",
        "registry_id",
        "dataset",
        "seed",
        "checkpoint_path",
        "validity_status",
        "missing_evaluation",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            for eval_name in row["missing_evaluations"]:
                writer.writerow(
                    {
                        "wandb_run_id": row["wandb_run_id"],
                        "wandb_run_name": row["wandb_run_name"],
                        "internal_run_id": row["internal_run_id"],
                        "canonical_run_id": row["canonical_run_id"],
                        "thesis_label": row["thesis_label"],
                        "display_label": row["display_label"],
                        "latex_label": row["latex_label"],
                        "intervention_group": row["intervention_group"],
                        "reference_label": row["reference_label"],
                        "label_kind": row["label_kind"],
                        "include_in_main_results": row["include_in_main_results"],
                        "include_in_sweep_figures": row["include_in_sweep_figures"],
                        "sweep_display_label": row["sweep_display_label"],
                        "sweep_index": row["sweep_index"],
                        "is_excluded": row["is_excluded"],
                        "exclude_reason": row["exclude_reason"],
                        "registry_id": row["registry_id"],
                        "dataset": row["dataset"],
                        "seed": row["seed"],
                        "checkpoint_path": row["checkpoint_path"],
                        "validity_status": row["validity_status"],
                        "missing_evaluation": eval_name,
                    }
                )


def build_wide_rows(base_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wide: dict[str, dict[str, Any]] = {row["wandb_run_id"]: dict(row) for row in base_rows}
    summary_values: dict[str, dict[str, float]] = defaultdict(dict)
    raw_summary_values: dict[str, dict[str, float]] = defaultdict(dict)

    for row in summary_rows:
        run_id = row["wandb_run_id"]
        summary_values[run_id][f"metric/{row['metric']}"] = row["value"]
        raw_summary_values[run_id][f"summary/{row['raw_metric']}"] = row["value"]

    for run_id, values in summary_values.items():
        wide[run_id].update(values)
    for run_id, values in raw_summary_values.items():
        wide[run_id].update(values)
    return list(wide.values())


def preserve_existing_nonempty_cells(wide_df: Any, wide_path: Path) -> Any:
    """Fill blanks in a fresh wide export from the previous CSV for the same W&B run.

    W&B summaries can occasionally omit a field that was present in an earlier export.
    This keeps existing non-empty cells instead of replacing them with blanks while still
    allowing new non-empty values to update the table.
    """
    if not wide_path.exists() or wide_df.empty or "wandb_run_id" not in wide_df.columns:
        return wide_df

    import pandas as pd

    old_df = pd.read_csv(wide_path, dtype=object, keep_default_na=False)
    if old_df.empty or "wandb_run_id" not in old_df.columns:
        return wide_df

    merged = wide_df.copy()
    for col in old_df.columns:
        if col not in merged.columns:
            merged[col] = ""

    old_by_run = old_df.set_index("wandb_run_id", drop=False)
    for idx, row in merged.iterrows():
        run_id = row.get("wandb_run_id")
        if is_missing(run_id) or run_id not in old_by_run.index:
            continue
        old_row = old_by_run.loc[run_id]
        if hasattr(old_row, "iloc") and getattr(old_row, "ndim", 1) > 1:
            old_row = old_row.iloc[-1]
        for col in old_df.columns:
            if is_missing(merged.at[idx, col]) and not is_missing(old_row.get(col)):
                merged.at[idx, col] = old_row.get(col)
    return merged


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    if args.limit is not None:
        runs = runs[: args.limit]

    label_map = load_thesis_label_map()
    checkpoint_by_wandb_id, checkpoint_by_identity = discover_checkpoints(args.results_root)
    runs_with_identity = [(run, run_identity(run)) for run in runs]
    present_internal_ids = {
        identity["internal_run_id"]
        for _, identity in runs_with_identity
        if identity["internal_run_id"] is not None
    }

    base_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    summary_metric_rows: list[dict[str, Any]] = []

    for run, identity in sorted(runs_with_identity, key=lambda item: (getattr(item[0], "name", "") or "", item[0].id)):
        config = dict(run.config or {})
        summary = dict(run.summary._json_dict)
        identity_key = (identity["registry_id"], identity["dataset"], identity["seed"])
        checkpoint_path = checkpoint_by_wandb_id.get(run.id)
        if checkpoint_path is None and all(x is not None for x in identity_key):
            checkpoint_path = checkpoint_by_identity.get(identity_key)  # type: ignore[arg-type]

        label_info = label_map.resolve(identity["internal_run_id"], present_ids=present_internal_ids)
        label_fields = (
            label_info.as_dict()
            if label_info is not None
            else {
                col: False
                if col
                in {
                    "is_label_alias",
                    "is_superseded",
                    "include_in_main_results",
                    "include_in_sweep_figures",
                    "is_excluded",
                }
                else ""
                for col in LABEL_COLUMNS
            }
        )
        if label_info is None and identity["internal_run_id"] is not None:
            label_fields["internal_run_id"] = identity["internal_run_id"]

        validity_reasons = []
        if identity["registry_id"] is None or identity["dataset"] is None or identity["seed"] is None:
            validity_reasons.append("missing_identity")
        if label_info is None:
            validity_reasons.append("unmapped_internal_run_id")
        elif label_info.is_excluded:
            validity_reasons.append(f"excluded:{label_info.label_kind}")
        elif label_info.is_superseded:
            validity_reasons.append(f"superseded_by:{label_info.superseded_by}")
        if not checkpoint_path:
            validity_reasons.append("missing_checkpoint")
        if run.state not in {"finished", "crashed", "failed", "running"}:
            validity_reasons.append(f"wandb_state:{run.state}")
        if run.state in {"crashed", "failed"}:
            validity_reasons.append(f"wandb_state:{run.state}")
        invalid_reasons = [
            reason
            for reason in validity_reasons
            if not reason.startswith("superseded_by:") and not reason.startswith("excluded:")
        ]
        validity_status = (
            "invalid"
            if invalid_reasons
            else "excluded"
            if label_info is not None and label_info.is_excluded
            else "superseded"
            if label_info is not None and label_info.is_superseded
            else "valid"
        )

        run_meta = {
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
            "wandb_state": run.state,
            "wandb_url": run.url,
            "created_at": str(run.created_at),
            **label_fields,
            "registry_id": identity["registry_id"],
            "dataset": identity["dataset"],
            "seed": identity["seed"],
            "validity_status": validity_status,
            "validity_reasons": validity_reasons,
        }
        flat_config = flatten("config", config)
        base_row = {
            **run_meta,
            "checkpoint_path": checkpoint_path or "",
            **flat_config,
        }
        if "config/run_id" not in base_row and identity["registry_id"] is not None:
            base_row["config/run_id"] = identity["registry_id"]
        if "config/dataset" not in base_row and identity["dataset"] is not None:
            base_row["config/dataset"] = identity["dataset"]
        if "config/seed" not in base_row and identity["seed"] is not None:
            base_row["config/seed"] = identity["seed"]
        if "config/unfreeze_layers" not in base_row:
            unfreeze_layers = first_config_value(
                config,
                "unfreeze_layers",
                "model/unfreeze_vision_layers",
                "model.unfreeze_vision_layers",
            )
            if unfreeze_layers is None and identity["registry_id"] is not None:
                unfreeze_layers = registry_unfreeze_layers().get(str(identity["registry_id"]))
            if unfreeze_layers is not None:
                base_row["config/unfreeze_layers"] = unfreeze_layers
        base_rows.append(base_row)

        summary_rows = metric_rows_for_mapping(summary, source="summary", run_meta=run_meta)
        summary_metric_rows.extend(summary_rows)
        all_metric_rows.extend(summary_rows)
        if args.include_history:
            all_metric_rows.extend(history_rows(run, run_meta, args.history_samples))

        summary_metric_names = {row["metric"] for row in summary_rows}
        completed, missing = completion_status(summary_metric_names, identity["dataset"])

        manifest_rows.append(
            {
                **run_meta,
                "checkpoint_path": checkpoint_path or "",
                "validity_status": validity_status,
                "validity_reasons": validity_reasons,
                "completed_evaluations": completed,
                "missing_evaluations": missing,
            }
        )

    long_df = pd.DataFrame(all_metric_rows)
    # sweep_index is float for mapped runs but "" for unmapped fallbacks; coerce to a
    # single numeric dtype so the parquet writer does not choke on the mixed object column.
    if "sweep_index" in long_df.columns:
        long_df["sweep_index"] = pd.to_numeric(long_df["sweep_index"], errors="coerce")
    wide_rows = build_wide_rows(base_rows, summary_metric_rows)
    wide_df = pd.DataFrame(wide_rows)

    manifest_path = args.out_dir / "manifest.jsonl"
    long_path = args.out_dir / "clean_results_long.parquet"
    wide_path = args.out_dir / "clean_results_wide.csv"
    missing_path = args.out_dir / "missing_evaluations.csv"

    wide_df = preserve_existing_nonempty_cells(wide_df, wide_path)

    write_jsonl(manifest_path, manifest_rows)
    if long_df.empty:
        long_df = pd.DataFrame(
            columns=[
                "wandb_run_id",
                "wandb_run_name",
                "internal_run_id",
                "canonical_run_id",
                "thesis_label",
                "display_label",
                "latex_label",
                "intervention_group",
                "reference_label",
                "label_kind",
                "include_in_main_results",
                "include_in_sweep_figures",
                "sweep_display_label",
                "sweep_index",
                "is_excluded",
                "exclude_reason",
                "is_label_alias",
                "alias_of",
                "superseded_by",
                "is_superseded",
                "registry_id",
                "dataset",
                "seed",
                "validity_status",
                "validity_reasons",
                "source",
                "step",
                "epoch",
                "raw_metric",
                "metric",
                "family",
                "split",
                "benchmark",
                "direction",
                "k",
                "value",
            ]
        )
    long_df.to_parquet(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    write_missing_csv(missing_path, manifest_rows)

    print(f"Wrote {manifest_path}")
    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {missing_path}")
    print(f"Runs exported: {len(manifest_rows)}")
    print(f"Metric rows exported: {len(long_df)}")


if __name__ == "__main__":
    main()
