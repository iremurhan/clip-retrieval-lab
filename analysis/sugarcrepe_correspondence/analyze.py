#!/usr/bin/env python3
"""Retrieval-vs-SugarCrepe correspondence analysis.

The script can consume a W&B summary CSV exported by scripts/util/export_wandb.py,
but it also falls back to checked-in result CSVs and local training logs.
It intentionally uses only the Python standard library so the analysis is
re-runnable in minimal cluster/login-node environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import glob
import math
import os
import random
import re
import statistics
import struct
import time
import zlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "data"
DEFAULT_LOG_ROOT = Path("/users/beyza.urhan/experiments/results")
OUT_ROOT = ROOT / "analysis" / "sugarcrepe_correspondence"

SC_CATEGORIES = [
    "add_att",
    "add_obj",
    "replace_att",
    "replace_obj",
    "replace_rel",
    "swap_att",
    "swap_obj",
]

SC_LABELS = {
    "add_att": "add-attribute",
    "add_obj": "add-object",
    "replace_att": "replace-attribute",
    "replace_obj": "replace-object",
    "replace_rel": "replace-relation",
    "swap_att": "swap-attribute",
    "swap_obj": "swap-object",
}

RUN_ORDER = [
    "B0plus",
    "B0_uf5",
    "B0_uf6",
    "B0_uf7",
    "B0_proj1024",
    "B1",
    "B2",
    "B4",
    "B5a_seg_spatial",
    "B5b_seg_semantic",
    "B5c_seg_continuous",
    "B5d_multistream_gate",
    "B5d_multistream_crossattn",
    "B5d_multistream_concat",
    "B5e_sam_skip",
    "BLIP_TEXT",
]

CAPACITY_RUNS = {"B0plus", "B0_uf5", "B0_uf6", "B0_uf7", "B0_proj1024"}


def ref_for(run_id: str) -> str:
    return "B0" if run_id in CAPACITY_RUNS else "B0plus"


def canon_run_id(run_id: str) -> str:
    if run_id == "B0plus_fixed":
        return "B0plus"
    if run_id.startswith("B5a_seg_spatial_flickr"):
        return "B5a_seg_spatial"
    if run_id.startswith("B5b_seg_semantic_flickr"):
        return "B5b_seg_semantic"
    return run_id


def fnum(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if math.isnan(value_f):
        return None
    return value_f


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concord = discord = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                ties_x += 1
            elif sy == 0:
                ties_y += 1
            elif sx == sy:
                concord += 1
            else:
                discord += 1
    denom = math.sqrt((concord + discord + ties_x) * (concord + discord + ties_y))
    return (concord - discord) / denom if denom else float("nan")


def bootstrap_ci(rows: list[dict], x_col: str, y_col: str, fn, n_boot: int, seed: int) -> tuple[float, float]:
    eligible = [r for r in rows if int(r["n_seeds"]) >= 2 and r.get(x_col) is not None and r.get(y_col) is not None]
    if len(eligible) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    n = len(eligible)
    for _ in range(n_boot):
        sample = [eligible[rng.randrange(n)] for _ in range(n)]
        x = [float(r[x_col]) for r in sample]
        y = [float(r[y_col]) for r in sample]
        val = fn(x, y)
        if not math.isnan(val):
            vals.append(val)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def pct_from_summary(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 1.5 else value


def add_metric(store: dict, run_id: str, dataset: str, seed: str, metric: str, value: float | None) -> None:
    if value is None:
        return
    run_id = canon_run_id(run_id)
    key = (run_id, dataset, str(seed))
    store[key][metric] = pct_from_summary(value)


def parse_run_name(name: str) -> tuple[str | None, str | None, str | None]:
    m = re.match(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)(?:_\d+)?$", name)
    if not m:
        return None, None, None
    dataset = "flickr30k" if m.group("dataset") == "flickr" else m.group("dataset")
    return canon_run_id(m.group("run")), dataset, m.group("seed")


def load_wandb_csv(path: Path, retrieval: dict, sugar: dict) -> None:
    if not path or not path.exists():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("state") and row.get("state") != "finished":
                continue
            run_id = row.get("config/run_id") or None
            dataset = row.get("config/dataset") or None
            seed = row.get("config/seed") or row.get("config/training/seed") or None
            if not run_id or not dataset or not seed:
                run_id, dataset, seed = parse_run_name(row.get("name", ""))
            if not run_id or not dataset or not seed:
                continue
            dataset = "flickr30k" if dataset == "flickr" else dataset
            if dataset == "coco":
                mapping = {
                    "r1_i2t": "summary/test/coco_5k_r1_i2t",
                    "r1_t2i": "summary/test/coco_5k_r1_t2i",
                    "r5_i2t": "summary/test/coco_5k_r5_i2t",
                    "r5_t2i": "summary/test/coco_5k_r5_t2i",
                    "r10_i2t": "summary/test/coco_5k_r10_i2t",
                    "r10_t2i": "summary/test/coco_5k_r10_t2i",
                    "eccv_map_at_r_i2t": "summary/test/eccv_map_at_r_i2t",
                    "eccv_map_at_r_t2i": "summary/test/eccv_map_at_r_t2i",
                    "eccv_rprecision_i2t": "summary/test/eccv_rprecision_i2t",
                    "eccv_rprecision_t2i": "summary/test/eccv_rprecision_t2i",
                }
            else:
                mapping = {
                    "r1_i2t": "summary/test/r1_i2t",
                    "r1_t2i": "summary/test/r1_t2i",
                    "r5_i2t": "summary/test/r5_i2t",
                    "r5_t2i": "summary/test/r5_t2i",
                    "r10_i2t": "summary/test/r10_i2t",
                    "r10_t2i": "summary/test/r10_t2i",
                }
            for metric, col in mapping.items():
                add_metric(retrieval, run_id, dataset, seed, metric, fnum(row.get(col)))
            for cat in SC_CATEGORIES:
                add_metric(sugar, run_id, dataset, seed, f"sc_{cat}", fnum(row.get(f"summary/sugarcrepe/{cat}")))
            overall = fnum(row.get("summary/sugarcrepe/macro_avg"))
            if overall is None:
                overall = fnum(row.get("summary/sugarcrepe/overall"))
            add_metric(sugar, run_id, dataset, seed, "sc_overall", overall)


def parse_wandb_config(path: Path) -> tuple[str | None, str | None, str | None]:
    """Parse W&B's simple `key: {value: ...}` config.yaml without PyYAML."""
    wanted = {"run_id", "dataset", "seed"}
    found: dict[str, str] = {}
    current = None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.rstrip()
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1]
            current = key if key in wanted else None
            continue
        if current and line.strip().startswith("value:"):
            value = line.split("value:", 1)[1].strip().strip('"').strip("'")
            found[current] = value
            current = None
    return canon_run_id(found.get("run_id")) if found.get("run_id") else None, found.get("dataset"), found.get("seed")


def load_local_wandb_files(log_root: Path, retrieval: dict, sugar: dict) -> None:
    """Read retained local W&B summaries/configs, if present."""
    if not log_root.exists():
        return
    for name in glob.glob(str(log_root / "**" / "wandb-summary.json"), recursive=True):
        summary_path = Path(name)
        config_path = summary_path.with_name("config.yaml")
        if not config_path.exists():
            continue
        run_id, dataset, seed = parse_wandb_config(config_path)
        if not run_id or not dataset or not seed:
            continue
        dataset = "flickr30k" if dataset == "flickr" else dataset
        try:
            summary = json.loads(summary_path.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        if dataset == "coco":
            mapping = {
                "r1_i2t": ["test/coco_5k_r1_i2t", "test/r1_i2t"],
                "r1_t2i": ["test/coco_5k_r1_t2i", "test/r1_t2i"],
                "r5_i2t": ["test/coco_5k_r5_i2t", "test/r5_i2t"],
                "r5_t2i": ["test/coco_5k_r5_t2i", "test/r5_t2i"],
                "r10_i2t": ["test/coco_5k_r10_i2t", "test/r10_i2t"],
                "r10_t2i": ["test/coco_5k_r10_t2i", "test/r10_t2i"],
            }
        else:
            mapping = {
                "r1_i2t": ["test/r1_i2t"],
                "r1_t2i": ["test/r1_t2i"],
                "r5_i2t": ["test/r5_i2t"],
                "r5_t2i": ["test/r5_t2i"],
                "r10_i2t": ["test/r10_i2t"],
                "r10_t2i": ["test/r10_t2i"],
            }
        mapping.update({
            "eccv_map_at_r_i2t": ["test/eccv_map_at_r_i2t"],
            "eccv_map_at_r_t2i": ["test/eccv_map_at_r_t2i"],
            "eccv_rprecision_i2t": ["test/eccv_rprecision_i2t"],
            "eccv_rprecision_t2i": ["test/eccv_rprecision_t2i"],
        })
        for metric, keys in mapping.items():
            for key in keys:
                if key in summary:
                    add_metric(retrieval, run_id, dataset, seed, metric, fnum(summary.get(key)))
                    break
        for cat in SC_CATEGORIES:
            add_metric(sugar, run_id, dataset, seed, f"sc_{cat}", fnum(summary.get(f"sugarcrepe/{cat}")))
        overall = fnum(summary.get("sugarcrepe/macro_avg"))
        if overall is None:
            overall = fnum(summary.get("sugarcrepe/overall"))
        add_metric(sugar, run_id, dataset, seed, "sc_overall", overall)


def load_retrieval_export(path: Path, retrieval_agg: dict) -> None:
    if not path.exists():
        return
    by_key = defaultdict(dict)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (canon_run_id(row["run_id"]), row["dataset"])
            metric = f"r{row['k']}_{row['direction']}"
            by_key[key][metric] = fnum(row.get("mean"))
            by_key[key][f"{metric}_std"] = fnum(row.get("std"))
            by_key[key]["n_seeds"] = max(int(row.get("n_seeds") or 0), int(by_key[key].get("n_seeds", 0)))
    retrieval_agg.update(by_key)


def load_sugar_export(path: Path, sugar_agg: dict) -> None:
    if not path.exists():
        return
    temp = defaultdict(dict)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = canon_run_id(row["run_id"])
            dataset = row.get("dataset") or "coco"
            cat = row["category"]
            metric = "sc_overall" if cat == "overall" else f"sc_{cat}"
            temp[(run_id, dataset)][metric] = fnum(row.get("mean"))
            temp[(run_id, dataset)][f"{metric}_std"] = fnum(row.get("std"))
            temp[(run_id, dataset)]["n_seeds"] = max(int(row.get("n_seeds") or 0), int(temp[(run_id, dataset)].get("n_seeds", 0)))
    sugar_agg.update(temp)


def infer_from_path(path: Path) -> tuple[str | None, str | None, str | None]:
    parts = path.parts
    dataset = None
    for p in parts:
        if p in {"coco", "flickr30k"}:
            dataset = p
    seed = None
    for p in parts:
        m = re.match(r"\d+_s(\d+)$", p)
        if m:
            seed = m.group(1)
    run_id = None
    for p in parts:
        if re.match(r"\d+_", p):
            candidate = re.sub(r"^\d+_", "", p)
            candidate = re.sub(r"_s\d+$", "", candidate)
            if candidate and candidate not in {"Results"}:
                run_id = candidate
    if path.parent.name.startswith("B"):
        parsed, ds, sd = parse_run_name(path.parent.name)
        run_id = parsed or run_id
        dataset = ds or dataset
        seed = sd or seed
    return canon_run_id(run_id) if run_id else None, dataset, seed


def load_training_logs(log_root: Path, retrieval: dict, sugar: dict) -> None:
    if not log_root.exists():
        return
    sc_re = re.compile(r"sugarcrepe/(add_att|add_obj|replace_att|replace_obj|replace_rel|swap_att|swap_obj|macro_avg|overall):\s+([0-9.]+)")
    test_re = re.compile(r"test/(coco_5k_)?r(1|5|10)_(i2t|t2i):\s+([0-9.]+)")
    eccv_re = re.compile(r"test/(eccv_map_at_r|eccv_rprecision)_(i2t|t2i):\s+([0-9.]+)")
    for name in glob.glob(str(log_root / "**" / "training.log"), recursive=True):
        path = Path(name)
        run_id, dataset, seed = infer_from_path(path)
        if not run_id or not dataset:
            continue
        if seed is None:
            run_id2, dataset2, seed2 = parse_run_name(path.parent.name)
            run_id = run_id2 or run_id
            dataset = dataset2 or dataset
            seed = seed2 or "unknown"
        text = path.read_text(errors="ignore")
        for m in test_re.finditer(text):
            coco_prefix, k, direction, value = m.groups()
            if dataset == "coco" and not coco_prefix:
                continue
            if dataset != "coco" and coco_prefix:
                continue
            add_metric(retrieval, run_id, dataset, seed, f"r{k}_{direction}", fnum(value))
        for m in eccv_re.finditer(text):
            family, direction, value = m.groups()
            add_metric(retrieval, run_id, dataset, seed, f"{family}_{direction}", fnum(value))
        for m in sc_re.finditer(text):
            cat, value = m.groups()
            metric = "sc_overall" if cat in {"macro_avg", "overall"} else f"sc_{cat}"
            add_metric(sugar, run_id, dataset, seed, metric, fnum(value))


def aggregate_seed_store(store: dict) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for (run_id, dataset, _seed), metrics in store.items():
        for metric, value in metrics.items():
            grouped[(run_id, dataset)][metric].append(value)
    out = {}
    for key, values_by_metric in grouped.items():
        rec = {}
        n_by_metric = []
        for metric, values in values_by_metric.items():
            rec[metric] = mean(values)
            rec[f"{metric}_std"] = sample_std(values)
            n_by_metric.append(len(values))
        rec["n_seeds"] = max(n_by_metric) if n_by_metric else 0
        out[key] = rec
    return out


def merge_agg(primary: dict, fallback: dict) -> dict:
    merged = {k: dict(v) for k, v in fallback.items()}
    for key, values in primary.items():
        merged.setdefault(key, {}).update({k: v for k, v in values.items() if v is not None})
    return merged


def build_delta_rows(retrieval_agg: dict, sugar_agg: dict) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    all_runs = [r for r in RUN_ORDER if r not in {"B0"}]
    for dataset in ["flickr30k", "coco"]:
        for run_id in all_runs:
            ref_id = ref_for(run_id)
            key = (run_id, dataset)
            ref_key = (ref_id, dataset)
            if key not in retrieval_agg or ref_key not in retrieval_agg:
                continue
            if key not in sugar_agg or ref_key not in sugar_agg:
                missing.append(f"{run_id}/{dataset}: missing SugarCrepe for intervention or reference")
                continue
            r = retrieval_agg[key]
            rr = retrieval_agg[ref_key]
            s = sugar_agg[key]
            sr = sugar_agg[ref_key]
            needed = ["r1_i2t", "r1_t2i", "r5_i2t", "r5_t2i", "r10_i2t", "r10_t2i"]
            if any(r.get(m) is None or rr.get(m) is None for m in needed):
                missing.append(f"{run_id}/{dataset}: missing retrieval")
                continue
            if s.get("sc_overall") is None or sr.get("sc_overall") is None:
                missing.append(f"{run_id}/{dataset}: missing SugarCrepe overall")
                continue
            row = {
                "intervention_id": run_id,
                "reference_id": ref_id,
                "dataset": dataset,
                "n_seeds": min(int(r.get("n_seeds", 0)), int(s.get("n_seeds", 0))),
                "single_seed": "yes" if min(int(r.get("n_seeds", 0)), int(s.get("n_seeds", 0))) < 2 else "no",
                "delta_R@1_I2T": r["r1_i2t"] - rr["r1_i2t"],
                "delta_R@1_T2I": r["r1_t2i"] - rr["r1_t2i"],
                "delta_R@5_avg": ((r["r5_i2t"] + r["r5_t2i"]) - (rr["r5_i2t"] + rr["r5_t2i"])) / 2.0,
                "delta_R@10_avg": ((r["r10_i2t"] + r["r10_t2i"]) - (rr["r10_i2t"] + rr["r10_t2i"])) / 2.0,
                "delta_R@1_avg": ((r["r1_i2t"] + r["r1_t2i"]) - (rr["r1_i2t"] + rr["r1_t2i"])) / 2.0,
                "delta_sugarcrepe_overall": s["sc_overall"] - sr["sc_overall"],
            }
            for cat in SC_CATEGORIES:
                metric = f"sc_{cat}"
                row[f"delta_sugarcrepe_{cat}"] = None
                if s.get(metric) is not None and sr.get(metric) is not None:
                    row[f"delta_sugarcrepe_{cat}"] = s[metric] - sr[metric]
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], RUN_ORDER.index(r["intervention_id"]) if r["intervention_id"] in RUN_ORDER else 999))
    return rows, missing


def corr_table(rows: list[dict], n_boot: int) -> list[dict]:
    out = []
    pairs = [
        ("delta_R@1_T2I", "delta_sugarcrepe_overall", "R@1_T2I vs SugarCrepe overall"),
        ("delta_R@1_I2T", "delta_sugarcrepe_overall", "R@1_I2T vs SugarCrepe overall"),
        ("delta_R@1_avg", "delta_sugarcrepe_overall", "R@1_avg vs SugarCrepe overall"),
    ]
    for dataset in ["flickr30k", "coco"]:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        for x_col, y_col, label in pairs:
            valid = [r for r in ds_rows if r.get(x_col) is not None and r.get(y_col) is not None]
            x = [float(r[x_col]) for r in valid]
            y = [float(r[y_col]) for r in valid]
            for name, fn in [("pearson", pearson), ("spearman", spearman), ("kendall_tau", kendall_tau)]:
                val = fn(x, y) if len(valid) >= 2 else float("nan")
                lo, hi = bootstrap_ci(valid, x_col, y_col, fn, n_boot, seed=13 + len(out))
                out.append({
                    "dataset": dataset,
                    "comparison": label,
                    "statistic": name,
                    "n_interventions": len(valid),
                    "n_ci_interventions": sum(1 for r in valid if int(r["n_seeds"]) >= 2),
                    "estimate": val,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
    return out


def subcategory_corrs(rows: list[dict]) -> list[dict]:
    out = []
    for dataset in ["flickr30k", "coco"]:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        for cat in SC_CATEGORIES:
            y_col = f"delta_sugarcrepe_{cat}"
            valid = [r for r in ds_rows if r.get("delta_R@1_T2I") is not None and r.get(y_col) is not None]
            val = spearman([float(r["delta_R@1_T2I"]) for r in valid], [float(r[y_col]) for r in valid]) if len(valid) >= 2 else float("nan")
            out.append({
                "dataset": dataset,
                "subcategory": SC_LABELS[cat],
                "spearman_delta_R1_T2I": val,
                "n_interventions": len(valid),
            })
    return out


def fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.6f}"
    return str(x)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fieldnames})


# --- tiny PNG renderer ------------------------------------------------------

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


class Canvas:
    def __init__(self, w: int, h: int, bg=(255, 255, 255)):
        self.w = w
        self.h = h
        self.px = [bg] * (w * h)

    def set(self, x: int, y: int, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y * self.w + x] = c

    def line(self, x0, y0, x1, y1, c):
        x0 = int(round(x0)); y0 = int(round(y0)); x1 = int(round(x1)); y1 = int(round(y1))
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set(x0, y0, c)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x0 += sx
            if e2 <= dx:
                err += dx; y0 += sy

    def rect(self, x0, y0, x1, y1, c, fill=False):
        if fill:
            for y in range(int(y0), int(y1) + 1):
                for x in range(int(x0), int(x1) + 1):
                    self.set(x, y, c)
        else:
            self.line(x0, y0, x1, y0, c); self.line(x1, y0, x1, y1, c)
            self.line(x1, y1, x0, y1, c); self.line(x0, y1, x0, y0, c)

    def circle(self, cx, cy, r, c):
        cx = int(cx); cy = int(cy); r = int(r)
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.set(x, y, c)

    def text(self, x, y, text, c=(20, 20, 20), scale=1):
        x0 = int(x)
        for ch in text.upper():
            glyph = FONT.get(ch, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        for sy in range(scale):
                            for sx in range(scale):
                                self.set(x0 + gx * scale + sx, int(y) + gy * scale + sy, c)
            x0 += 6 * scale

    def save_png(self, path: Path):
        raw = bytearray()
        for y in range(self.h):
            raw.append(0)
            for x in range(self.w):
                raw.extend(self.px[y * self.w + x])
        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def nice_range(values: list[float]) -> tuple[float, float]:
    lo = min(values + [0.0]); hi = max(values + [0.0])
    if lo == hi:
        lo -= 1.0; hi += 1.0
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def draw_scatter(rows: list[dict], path: Path) -> None:
    c = Canvas(1400, 700)
    panel_w = 620
    margin_l = 80
    margin_t = 80
    plot_w = 500
    plot_h = 470
    colors = {"flickr30k": (39, 111, 191), "coco": (197, 83, 44)}
    for pi, dataset in enumerate(["flickr30k", "coco"]):
        ds = [r for r in rows if r["dataset"] == dataset]
        ox = 40 + pi * 680
        oy = 40
        c.text(ox + 190, oy + 5, dataset, scale=2)
        if not ds:
            c.text(ox + 160, oy + 250, "NO COMPLETE ROWS", scale=2)
            continue
        xs = [float(r["delta_R@1_T2I"]) for r in ds]
        ys = [float(r["delta_sugarcrepe_overall"]) for r in ds]
        xmin, xmax = nice_range(xs); ymin, ymax = nice_range(ys)
        def sx(x): return ox + margin_l + (x - xmin) / (xmax - xmin) * plot_w
        def sy(y): return oy + margin_t + plot_h - (y - ymin) / (ymax - ymin) * plot_h
        c.rect(ox + margin_l, oy + margin_t, ox + margin_l + plot_w, oy + margin_t + plot_h, (0, 0, 0))
        if xmin <= 0 <= xmax:
            c.line(sx(0), oy + margin_t, sx(0), oy + margin_t + plot_h, (190, 190, 190))
        if ymin <= 0 <= ymax:
            c.line(ox + margin_l, sy(0), ox + margin_l + plot_w, sy(0), (190, 190, 190))
        c.text(ox + margin_l + 285, oy + margin_t + 20, "HELPS BOTH", (80, 120, 80), 1)
        c.text(ox + margin_l + 25, oy + margin_t + 20, "COMPOSITION ONLY", (80, 120, 80), 1)
        c.text(ox + margin_l + 300, oy + margin_t + plot_h - 25, "RETRIEVAL ONLY", (150, 80, 80), 1)
        c.text(ox + margin_l + 35, oy + margin_t + plot_h - 25, "HURTS BOTH", (150, 80, 80), 1)
        for r in ds:
            x = sx(float(r["delta_R@1_T2I"])); y = sy(float(r["delta_sugarcrepe_overall"]))
            c.circle(x, y, 5, colors[dataset])
            c.text(x + 7, y - 4, r["intervention_id"].replace("B5d_multistream_", "B5d_").replace("_seg_", "_"), scale=1)
        c.text(ox + margin_l + 130, oy + margin_t + plot_h + 35, "DELTA R@1 T2I", scale=1)
        c.text(ox + 5, oy + margin_t + 210, "DELTA SUGARCREPE", scale=1)
    c.save_png(path)


def draw_heatmap(corrs: list[dict], path: Path) -> None:
    c = Canvas(1100, 420)
    cats = [SC_LABELS[x] for x in SC_CATEGORIES]
    datasets = ["flickr30k", "coco"]
    cell_w = 120; cell_h = 80
    x0 = 180; y0 = 100
    c.text(300, 30, "SPEARMAN: DELTA R@1 T2I VS SUGARCREPE SUBCATEGORY", scale=2)
    vals = {(r["dataset"], r["subcategory"]): r["spearman_delta_R1_T2I"] for r in corrs}
    for j, cat in enumerate(cats):
        c.text(x0 + j * cell_w + 8, y0 - 35, cat[:14], scale=1)
    for i, dataset in enumerate(datasets):
        c.text(40, y0 + i * cell_h + 28, dataset, scale=1)
        for j, cat in enumerate(cats):
            v = vals.get((dataset, cat), float("nan"))
            if math.isnan(v):
                color = (235, 235, 235)
                label = "NA"
            else:
                if v >= 0:
                    t = min(1.0, v)
                    color = (int(245 - 120 * t), int(245 - 55 * t), int(245 - 145 * t))
                else:
                    t = min(1.0, -v)
                    color = (int(245 - 30 * t), int(245 - 115 * t), int(245 - 115 * t))
                label = f"{v:+.2f}"
            x = x0 + j * cell_w; y = y0 + i * cell_h
            c.rect(x, y, x + cell_w - 2, y + cell_h - 2, color, fill=True)
            c.rect(x, y, x + cell_w - 2, y + cell_h - 2, (255, 255, 255))
            c.text(x + 35, y + 32, label, scale=1)
    c.text(250, 315, "GREEN POSITIVE  RED NEGATIVE  GRAY NOT ENOUGH COMPLETE ROWS", scale=1)
    c.save_png(path)


def summarize(rows: list[dict], corrs: list[dict], subcorrs: list[dict], missing: list[str]) -> str:
    lines = ["# Retrieval-SugarCrepe Correspondence Summary", ""]
    lines.append("Deltas are intervention minus its reference: B0 for capacity/B0plus rows and B0plus for loss, supervision, segment, multi-stream, and diagnostic rows. Rows with only one available seed are included in the tables and scatter, but excluded from bootstrap CI resampling.")
    lines.append("")
    for dataset in ["flickr30k", "coco"]:
        ds = [r for r in rows if r["dataset"] == dataset]
        if not ds:
            lines.append(f"For {dataset}, no complete intervention rows were available after requiring both retrieval and SugarCrepe metrics.")
            continue
        main = next((r for r in corrs if r["dataset"] == dataset and r["comparison"].startswith("R@1_T2I") and r["statistic"] == "spearman"), None)
        pear = next((r for r in corrs if r["dataset"] == dataset and r["comparison"].startswith("R@1_T2I") and r["statistic"] == "pearson"), None)
        helps_both = [r["intervention_id"] for r in ds if r["delta_R@1_T2I"] > 0 and r["delta_sugarcrepe_overall"] > 0]
        retrieval_only = [r["intervention_id"] for r in ds if r["delta_R@1_T2I"] > 0 and r["delta_sugarcrepe_overall"] <= 0]
        comp_only = [r["intervention_id"] for r in ds if r["delta_R@1_T2I"] <= 0 and r["delta_sugarcrepe_overall"] > 0]
        hurts_both = [r["intervention_id"] for r in ds if r["delta_R@1_T2I"] <= 0 and r["delta_sugarcrepe_overall"] <= 0]
        lines.append(
            f"For {dataset}, retrieval-composition correspondence is "
            f"Spearman {fmt(main['estimate']) if main else 'NA'} and Pearson {fmt(pear['estimate']) if pear else 'NA'} "
            f"between delta R@1 T2I and delta SugarCrepe overall across {len(ds)} complete interventions."
        )
        lines.append(f"Helps both: {', '.join(helps_both) if helps_both else 'none'}. Retrieval-only: {', '.join(retrieval_only) if retrieval_only else 'none'}. Composition-only: {', '.join(comp_only) if comp_only else 'none'}. Hurts both: {', '.join(hurts_both) if hurts_both else 'none'}.")
        sub = [r for r in subcorrs if r["dataset"] == dataset and not math.isnan(r["spearman_delta_R1_T2I"])]
        if sub:
            sub_sorted = sorted(sub, key=lambda r: r["spearman_delta_R1_T2I"], reverse=True)
            lines.append(f"The closest-tracking subcategory is {sub_sorted[0]['subcategory']} (rho={sub_sorted[0]['spearman_delta_R1_T2I']:.2f}); the most decoupled/negative is {sub_sorted[-1]['subcategory']} (rho={sub_sorted[-1]['spearman_delta_R1_T2I']:.2f}).")
        lines.append("")
    if missing:
        lines.append("Missingness note: interventions without both retrieval and SugarCrepe metrics were skipped rather than imputed. First missing cases: " + "; ".join(missing[:8]) + ("." if len(missing) <= 8 else "; ..."))
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-csv", type=Path, default=None, help="Optional runs_summary.csv from scripts/util/export_wandb.py")
    p.add_argument("--results-data", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--timestamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()

    out_dir = OUT_ROOT / args.timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_seed = defaultdict(dict)
    sugar_seed = defaultdict(dict)
    if args.wandb_csv:
        load_wandb_csv(args.wandb_csv, retrieval_seed, sugar_seed)
    load_local_wandb_files(args.log_root, retrieval_seed, sugar_seed)
    load_training_logs(args.log_root, retrieval_seed, sugar_seed)

    retrieval_from_seed = aggregate_seed_store(retrieval_seed)
    sugar_from_seed = aggregate_seed_store(sugar_seed)
    retrieval_export = {}
    sugar_export = {}
    load_retrieval_export(args.results_data / "04A_retrieval_grouped_bar_data.csv", retrieval_export)
    load_sugar_export(args.results_data / "01_sugarcrepe_interventions_data.csv", sugar_export)

    retrieval_agg = merge_agg(retrieval_from_seed, retrieval_export)
    sugar_agg = merge_agg(sugar_from_seed, sugar_export)

    rows, missing = build_delta_rows(retrieval_agg, sugar_agg)
    corr_rows = corr_table(rows, args.bootstrap)
    subcorr_rows = subcategory_corrs(rows)

    delta_fields = [
        "intervention_id", "reference_id", "dataset", "n_seeds", "single_seed",
        "delta_R@1_I2T", "delta_R@1_T2I", "delta_R@5_avg", "delta_R@10_avg",
        "delta_R@1_avg", "delta_sugarcrepe_overall",
    ] + [f"delta_sugarcrepe_{cat}" for cat in SC_CATEGORIES]
    write_csv(out_dir / "intervention_deltas.csv", rows, delta_fields)
    write_csv(out_dir / "correlations.csv", corr_rows, ["dataset", "comparison", "statistic", "n_interventions", "n_ci_interventions", "estimate", "ci95_low", "ci95_high"])
    write_csv(out_dir / "subcategory_spearman.csv", subcorr_rows, ["dataset", "subcategory", "spearman_delta_R1_T2I", "n_interventions"])
    (out_dir / "missing_metrics.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    draw_scatter(rows, out_dir / "quadrant_scatter.png")
    draw_heatmap(subcorr_rows, out_dir / "subcategory_heatmap.png")
    (out_dir / "summary.md").write_text(summarize(rows, corr_rows, subcorr_rows, missing))
    print(f"Wrote analysis outputs to {out_dir}")
    print(f"Complete intervention rows: {len(rows)}")


if __name__ == "__main__":
    main()
