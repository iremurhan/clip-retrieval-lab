from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.thesis_labels import apply_thesis_labels, load_thesis_label_map


ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/figures")
)
FIG_ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_FIG_ARTIFACT_ROOT", ARTIFACT_ROOT)
)
SAVE_FIG_DIR = Path(os.environ.get("SAVE_FIG_DIR", FIG_ARTIFACT_ROOT / "figures"))
SAVE_TABLE_DIR = Path(os.environ.get("SAVE_TABLE_DIR", FIG_ARTIFACT_ROOT / "tables"))
SAVE_DATA_DIR = Path(os.environ.get("SAVE_DATA_DIR", FIG_ARTIFACT_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ARTIFACT_ROOT / "cache"))
DEFAULT_CSV_PATH = Path(
    os.environ.get("CLEAN_RESULTS_WIDE_PATH", ARTIFACT_ROOT / "clean" / "clean_results_wide.csv")
)
EXCLUDE = ["B0v2", "B5_seg"]


def ensure_artifact_dirs() -> None:
    """Create the external artifact directory tree used by figure scripts."""
    for path in (
        ARTIFACT_ROOT,
        FIG_ARTIFACT_ROOT,
        FIG_ARTIFACT_ROOT / "cache",
        SAVE_FIG_DIR,
        SAVE_TABLE_DIR,
        SAVE_DATA_DIR,
        CACHE_DIR,
        CACHE_DIR / "mplconfig",
    ):
        path.mkdir(parents=True, exist_ok=True)


ensure_artifact_dirs()

SC_CATEGORIES = [
    ("add_att", "Add\nAttribute"),
    ("add_obj", "Add\nObject"),
    ("replace_att", "Replace\nAttribute"),
    ("replace_obj", "Replace\nObject"),
    ("replace_rel", "Replace\nRelation"),
    ("swap_att", "Swap\nAttribute"),
    ("swap_obj", "Swap\nObject"),
]

THESIS_LABEL_ORDER = [
    "Base-min",
    "Intra-Reg",
    "Unfreeze-5",
    "Unfreeze-6",
    "Unfreeze-7",
    "Proj-1024",
    "Loss-SigLIP",
    "HN-Syntactic",
    "Aux-ObjCls",
    "Seg-Spatial",
    "Seg-Semantic",
    "Seg-Geom",
    "Sam-Gate",
    "Sam-XAttn",
    "Sam-Concat",
    "Sam-Skip",
    "Text-BLIP",
]
RETRIEVAL_CONFIG_ORDER = THESIS_LABEL_ORDER

RETRIEVAL_DATASETS = {
    "flickr30k": {
        "label": "Flickr30K",
        "columns": {
            ("i2t", 1): "summary/test/r1_i2t",
            ("i2t", 5): "summary/test/r5_i2t",
            ("i2t", 10): "summary/test/r10_i2t",
            ("t2i", 1): "summary/test/r1_t2i",
            ("t2i", 5): "summary/test/r5_t2i",
            ("t2i", 10): "summary/test/r10_t2i",
        },
    },
    "coco": {
        "label": "COCO 5K",
        "columns": {
            ("i2t", 1): "summary/test/coco_5k_r1_i2t",
            ("i2t", 5): "summary/test/coco_5k_r5_i2t",
            ("i2t", 10): "summary/test/coco_5k_r10_i2t",
            ("t2i", 1): "summary/test/coco_5k_r1_t2i",
            ("t2i", 5): "summary/test/coco_5k_r5_t2i",
            ("t2i", 10): "summary/test/coco_5k_r10_t2i",
        },
    },
}

RETRIEVAL_DIRECTIONS = {"i2t": "I2T", "t2i": "T2I"}
RETRIEVAL_KS = [1, 5, 10]


def _copy_attrs(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    dst.attrs.update(src.attrs)
    return dst


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _string_values(values: Iterable[object]) -> list[str]:
    return sorted(str(value) for value in values if pd.notna(value))


def _invalid_only_missing_checkpoint(reasons: object) -> bool:
    """True when a run's only invalidating reason is a missing local checkpoint.

    The clean pipeline serializes validity_reasons as a JSON list string. A run with
    valid W&B metrics but no discoverable best_model.pth is still plottable, since the
    figures visualize already-logged metrics rather than re-evaluating checkpoints.
    """
    import ast
    import json

    if reasons is None or (isinstance(reasons, float) and pd.isna(reasons)):
        return False
    parsed = reasons
    if isinstance(reasons, str):
        text = reasons.strip()
        # The wide CSV may store the list as JSON ("[...]" with double quotes) or as a
        # Python repr ("['...']" with single quotes); accept both.
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                break
            except (ValueError, SyntaxError, TypeError):
                parsed = [reasons]
    if not isinstance(parsed, (list, tuple)):
        return False
    blocking = [
        str(r)
        for r in parsed
        if not str(r).startswith("superseded_by:") and not str(r).startswith("excluded:")
    ]
    return blocking == ["missing_checkpoint"]


def _ensure_internal_run_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "internal_run_id" in out.columns:
        return out
    for col in ("config/run_id", "registry_id", "run_id"):
        if col in out.columns:
            out["internal_run_id"] = out[col]
            return out
    raise KeyError("Expected one of 'internal_run_id', 'config/run_id', 'registry_id', or 'run_id'.")


def label_cache_frame(
    df: pd.DataFrame,
    *,
    id_col: str = "run_id",
    context: str = "diagnostic cache",
    drop_superseded: bool = True,
    result_scope: str = "main",
) -> pd.DataFrame:
    """Attach canonical thesis labels to a diagnostic cache keyed by raw run_id."""
    labeled = apply_thesis_labels(
        df,
        id_col=id_col,
        context=context,
        fail_on_unmapped=True,
        drop_superseded=drop_superseded,
    )
    return _filter_result_scope(labeled, result_scope)


def _filter_result_scope(df: pd.DataFrame, result_scope: str) -> pd.DataFrame:
    if result_scope not in {"main", "sweep", "all"}:
        raise ValueError(f"Unknown result_scope={result_scope!r}; expected 'main', 'sweep', or 'all'.")
    out = df.copy()
    if "is_excluded" in out.columns:
        out = out.loc[~out["is_excluded"].astype(bool)].copy()
    if result_scope == "main":
        out = out.loc[out["include_in_main_results"].astype(bool)].copy()
    elif result_scope == "sweep":
        out = out.loc[out["include_in_sweep_figures"].astype(bool)].copy()
    return out


def label_for_internal_run_id(run_id: object, *, field: str = "display_label") -> str:
    info = load_thesis_label_map().resolve(run_id)
    if info is None:
        raise ValueError(f"Unmapped internal run ID: {run_id}")
    return str(getattr(info, field))


def load_runs(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    exclude_list: list[str] | tuple[str, ...] | None = None,
    result_scope: str = "main",
    drop_superseded: bool = True,
) -> pd.DataFrame:
    """Load CSV, apply EXCLUDE filter, return DataFrame."""
    df = pd.read_csv(csv_path)
    df = _ensure_internal_run_id(df)
    if "config/run_id" not in df.columns:
        df["config/run_id"] = df["internal_run_id"]

    df = df[df["internal_run_id"].notna()].copy()
    excludes = EXCLUDE if exclude_list is None else list(exclude_list)
    excluded = _string_values(set(df["internal_run_id"]) & set(excludes))
    filtered = df[~df["internal_run_id"].isin(excludes)].copy()
    if "validity_status" in filtered.columns:
        invalid_statuses = {"invalid", "excluded"}
        if drop_superseded:
            invalid_statuses.add("superseded")
        drop_mask = filtered["validity_status"].isin(invalid_statuses)
        # A run flagged "invalid" only because its local checkpoint was not found still
        # carries valid W&B metrics, so keep it for plotting (we visualize metrics, not
        # re-evaluate checkpoints). Runs invalid for any other reason stay dropped.
        if "validity_reasons" in filtered.columns:
            keep_metric_only = filtered["validity_status"].eq("invalid") & filtered[
                "validity_reasons"
            ].map(_invalid_only_missing_checkpoint)
            drop_mask = drop_mask & ~keep_metric_only
        invalid = _string_values(filtered.loc[drop_mask, "internal_run_id"].unique())
        filtered = filtered.loc[~drop_mask].copy()
    else:
        invalid = []
    filtered = apply_thesis_labels(
        filtered,
        id_col="internal_run_id",
        context=f"figure input {csv_path}",
        fail_on_unmapped=True,
        drop_superseded=drop_superseded,
        warn_superseded=drop_superseded,
    )
    filtered = _filter_result_scope(filtered, result_scope)
    filtered.attrs["excluded_hard"] = excluded
    filtered.attrs["excluded_invalid"] = invalid
    return filtered


def filter_sugarcrepe_coco(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to coco rows with SugarCrepe results.

    Adds a unified 'sc_overall' column using macro_avg if present, else
    overall. Also computes per-category unified columns sc_<category>.
    """
    if "config/dataset" not in df.columns:
        raise KeyError("Missing required column: config/dataset")
    if "summary/sugarcrepe/overall" not in df.columns and "summary/sugarcrepe/macro_avg" not in df.columns:
        raise KeyError("Expected at least one SugarCrepe overall column.")

    working = df.copy()
    overall = _numeric_series(working, "summary/sugarcrepe/overall")
    macro_avg = _numeric_series(working, "summary/sugarcrepe/macro_avg")
    has_sugarcrepe = overall.notna() | macro_avg.notna()
    is_coco = working["config/dataset"].eq("coco")

    excluded_non_coco = _string_values(working.loc[has_sugarcrepe & ~is_coco, "config/run_id"].unique())
    excluded_no_sugarcrepe = _string_values(working.loc[is_coco & ~has_sugarcrepe, "config/run_id"].unique())

    filtered = working.loc[is_coco & has_sugarcrepe].copy()
    filtered["sc_overall"] = macro_avg.loc[filtered.index].combine_first(overall.loc[filtered.index])
    for category, _ in SC_CATEGORIES:
        filtered[f"sc_{category}"] = _numeric_series(filtered, f"summary/sugarcrepe/{category}")

    _copy_attrs(df, filtered)
    filtered.attrs["excluded_non_coco_with_sugarcrepe"] = excluded_non_coco
    filtered.attrs["excluded_coco_without_sugarcrepe"] = excluded_no_sugarcrepe
    return filtered


def split_by_baseline(
    df: pd.DataFrame,
    include_b0_in_interventions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (b0plus_df, b0_df) based on intra_img_weight and intra_txt_weight columns."""
    required = ["config/intra_img_weight", "config/intra_txt_weight"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        label_required = {"thesis_label", "reference_label"}
        if not label_required.issubset(df.columns):
            raise KeyError(f"Missing required columns: {missing}")

        b0plus_mask = df["thesis_label"].eq("Intra-Reg") | df["reference_label"].eq("Intra-Reg")
        b0_mask = df["thesis_label"].eq("Base-min") | (
            df["reference_label"].eq("Base-min") & ~df["thesis_label"].eq("Intra-Reg")
        )

        b0plus_df = df.loc[b0plus_mask].copy()
        b0_df = df.loc[b0_mask].copy()
        if include_b0_in_interventions:
            b0_only = df[df["thesis_label"].eq("Base-min")]
            b0plus_df = pd.concat([b0_only, b0plus_df], ignore_index=True)

        no_family = _string_values(df.loc[~(b0plus_mask | b0_mask), "display_label"].unique())
        for subset in (b0plus_df, b0_df):
            _copy_attrs(df, subset)
            subset.attrs["excluded_no_family"] = no_family
        return b0plus_df, b0_df

    img_weight = _numeric_series(df, "config/intra_img_weight")
    txt_weight = _numeric_series(df, "config/intra_txt_weight")
    b0plus_mask = img_weight.eq(1) & txt_weight.eq(1)
    b0_mask = img_weight.eq(0) & txt_weight.eq(0)

    b0plus_df = df.loc[b0plus_mask].copy()
    b0_df = df.loc[b0_mask].copy()
    if include_b0_in_interventions:
        b0_only = b0_df[b0_df["config/run_id"].eq("B0")]
        b0plus_df = pd.concat([b0_only, b0plus_df], ignore_index=True)

    no_family = _string_values(df.loc[~(b0plus_mask | b0_mask), "config/run_id"].unique())
    for subset in (b0plus_df, b0_df):
        _copy_attrs(df, subset)
        subset.attrs["excluded_no_family"] = no_family
    return b0plus_df, b0_df


def aggregate_by_config(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Group by canonical thesis label, return seed means/stds per value column."""
    if df.empty:
        columns = [
            "thesis_label",
            "run_id",
            "display_label",
            "latex_label",
            "intervention_group",
            "reference_label",
            "n_seeds",
        ]
        for col in value_cols:
            columns.extend([f"{col}_mean", f"{col}_std"])
        return pd.DataFrame(columns=columns)

    missing = [col for col in value_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing value columns: {missing}")

    group_cols = ["thesis_label", "display_label", "latex_label", "intervention_group", "reference_label"]
    missing_labels = [col for col in group_cols if col not in df.columns]
    if missing_labels:
        raise KeyError(f"Missing thesis label columns: {missing_labels}")

    grouped = df.groupby(group_cols, sort=False, dropna=False)
    mean_df = grouped[value_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped[value_cols].std(ddof=1, numeric_only=True).add_suffix("_std")
    n_df = grouped.size().rename("n_seeds")
    out = pd.concat([mean_df, std_df, n_df], axis=1).reset_index()
    out.insert(1, "run_id", out["thesis_label"])
    return out


def aggregate_seeds(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Backward-compatible aggregation used by older figure scripts."""
    group_cols = [
        "thesis_label",
        "display_label",
        "latex_label",
        "intervention_group",
        "reference_label",
        "config/dataset",
    ]
    grouped = df.groupby(group_cols, dropna=False)
    mean_df = grouped[value_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped[value_cols].std(ddof=1, numeric_only=True).add_suffix("_std")
    n_df = grouped.size().rename("n_seeds")
    out = pd.concat([mean_df, std_df, n_df], axis=1).reset_index()
    out.insert(1, "run_id", out["thesis_label"])
    return out


def setup_thesis_style() -> None:
    """Set matplotlib rcParams: serif font, no grid by default, thin spines, sensible figure DPI."""
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mplconfig"))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="white", context="paper")
    except Exception:
        plt.style.use("default")

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "axes.grid": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "0.2",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_palette(n_configs: int):
    """Return colorblind-safe palette of n_configs distinct colors."""
    import seaborn as sns

    return sns.color_palette("colorblind", n_colors=max(n_configs, 1))


# --- Shared single-seed visual convention -------------------------------------------
# One convention across thesis figures: single-seed (n=1) estimates get a "*"
# suffix on labels/text and a hollow marker on line plots.
SINGLE_SEED_FOOTNOTE = "* single seed (n=1)."


def single_seed_label(label: object, n_seeds: object) -> str:
    """Append the single-seed asterisk to a display label when n_seeds == 1."""
    text = str(label)
    try:
        n = int(n_seeds)
    except (TypeError, ValueError):
        return text
    return f"{text}*" if n == 1 else text


def single_seed_marker_kw(color) -> dict:
    """Scatter kwargs for a hollow single-seed point in line plots."""
    return {"facecolors": "white", "edgecolors": [color], "linewidths": 1.2}


def _to_percent(value: object) -> float:
    if pd.isna(value):
        return np.nan
    value = float(value)
    return value * 100.0 if abs(value) <= 1.5 else value


def sugarcrepe_aggregate_to_long(aggregate: pd.DataFrame, config_order: list[str]) -> pd.DataFrame:
    aggregate = aggregate.set_index("run_id", drop=False)

    rows = []
    for run_id in config_order:
        if run_id not in aggregate.index:
            continue
        run = aggregate.loc[run_id]
        for category, _ in [*SC_CATEGORIES, ("overall", "Overall")]:
            value_col = "sc_overall" if category == "overall" else f"sc_{category}"
            rows.append(
                {
                    "run_id": run_id,
                    "display_label": run.get("display_label", run_id),
                    "latex_label": run.get("latex_label", run_id),
                    "intervention_group": run.get("intervention_group", ""),
                    "reference_label": run.get("reference_label", ""),
                    "category": category,
                    "mean": _to_percent(run[f"{value_col}_mean"]),
                    "std": _to_percent(run[f"{value_col}_std"]),
                    "n_seeds": int(run["n_seeds"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "display_label",
            "latex_label",
            "intervention_group",
            "reference_label",
            "category",
            "mean",
            "std",
            "n_seeds",
        ],
    )


def retrieval_config_order(run_ids: Iterable[object]) -> list[str]:
    detected = [str(run_id) for run_id in run_ids if pd.notna(run_id)]
    ordered = [run_id for run_id in RETRIEVAL_CONFIG_ORDER if run_id in detected]
    extras = sorted(run_id for run_id in set(detected) if run_id not in RETRIEVAL_CONFIG_ORDER)
    return ordered + extras


def build_retrieval_data(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Build long-form retrieval data with per-config seed means and stds."""
    df = load_runs(csv_path, EXCLUDE)
    if "config/dataset" not in df.columns:
        raise KeyError("Missing required column: config/dataset")

    working = df.copy()
    working["dataset"] = working["config/dataset"].replace({"flickr": "flickr30k"})
    warnings: list[str] = []

    missing_columns = sorted(
        {
            col
            for spec in RETRIEVAL_DATASETS.values()
            for col in spec["columns"].values()
            if col not in working.columns
        }
    )
    if missing_columns:
        warnings.append("Missing source columns: " + ", ".join(missing_columns))
        for col in missing_columns:
            working[col] = np.nan

    source_cols = sorted({col for spec in RETRIEVAL_DATASETS.values() for col in spec["columns"].values()})
    has_any_retrieval = working[source_cols].apply(lambda col: pd.to_numeric(col, errors="coerce")).notna().any(axis=1)
    config_order = retrieval_config_order(working.loc[has_any_retrieval, "thesis_label"].unique())
    no_metric_runs = sorted(set(working["display_label"].dropna().astype(str)) - set(config_order))
    if no_metric_runs:
        warnings.append("Rows with no primary retrieval metrics excluded: " + ", ".join(no_metric_runs))

    suggested = set(RETRIEVAL_CONFIG_ORDER)
    extras = [run_id for run_id in config_order if run_id not in suggested]
    if extras:
        warnings.append("Additional configs appended after suggested order: " + ", ".join(extras))

    rows = []
    for dataset, spec in RETRIEVAL_DATASETS.items():
        dataset_df = working[working["dataset"].eq(dataset)].copy()
        present = set(dataset_df["thesis_label"].dropna().astype(str))
        for run_id in config_order:
            if run_id not in present:
                warnings.append(f"{spec['label']}: no rows for {run_id}")
                continue

            run_df = dataset_df[dataset_df["thesis_label"].eq(run_id)]
            label_row = run_df.iloc[0]
            dataset_cols = list(spec["columns"].values())
            has_dataset_metrics = run_df[dataset_cols].apply(lambda col: pd.to_numeric(col, errors="coerce")).notna().to_numpy().any()
            if not has_dataset_metrics:
                warnings.append(f"{spec['label']} {run_id}: no primary retrieval metrics")
                continue

            for direction in RETRIEVAL_DIRECTIONS:
                for k in RETRIEVAL_KS:
                    source_col = spec["columns"][(direction, k)]
                    raw = _numeric_series(run_df, source_col)
                    if "config/seed" in run_df.columns:
                        # Collapse duplicate W&B runs for the same seed before aggregating,
                        # so n_seeds reflects distinct seeds (not raw run rows).
                        per_seed = (
                            pd.DataFrame({"seed": run_df["config/seed"].values, "value": raw.values})
                            .dropna(subset=["value"])
                            .groupby("seed", dropna=False)["value"]
                            .mean()
                        )
                        values = per_seed.dropna().map(_to_percent)
                    else:
                        values = raw.dropna().map(_to_percent)
                    if values.empty:
                        warnings.append(f"{spec['label']} {run_id}: missing R@{k} {RETRIEVAL_DIRECTIONS[direction]}")
                        mean = np.nan
                        std = np.nan
                        n_seeds = 0
                    else:
                        mean = float(values.mean())
                        std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
                        n_seeds = int(len(values))

                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": spec["label"],
                            "run_id": run_id,
                            "display_label": label_row["display_label"],
                            "latex_label": label_row["latex_label"],
                            "intervention_group": label_row["intervention_group"],
                            "reference_label": label_row["reference_label"],
                            "config_order": config_order.index(run_id),
                            "direction": direction,
                            "direction_label": RETRIEVAL_DIRECTIONS[direction],
                            "k": k,
                            "metric": f"R@{k}",
                            "metric_label": f"R@{k} {RETRIEVAL_DIRECTIONS[direction]}",
                            "source_col": source_col,
                            "mean": mean,
                            "std": std,
                            "n_seeds": n_seeds,
                        }
                    )

    if not rows:
        raise ValueError("No retrieval rows found for Flickr30K or COCO.")

    data = pd.DataFrame(rows)
    data.attrs["config_order"] = config_order
    data.attrs["warnings"] = sorted(set(warnings))
    return data


# --- Intervention / capacity taxonomy (keyed on thesis_label_map intervention_group) ---
MAIN_INTERVENTION_GROUPS = {
    "loss_function",
    "hard_negatives",
    "auxiliary_object_classification",
    "segment_enrichment",
    "sam_fusion",
    "text_encoder",
}
CAPACITY_GROUPS = {"unfreezing_depth", "projection_capacity"}
BASELINE_GROUPS = {"baseline", "intra_regularization"}

# Canonical row order for the main-intervention figures (subset of THESIS_LABEL_ORDER).
MAIN_INTERVENTION_ORDER = [
    "Loss-SigLIP",
    "HN-Syntactic",
    "Aux-ObjCls",
    "Seg-Spatial",
    "Seg-Semantic",
    "Seg-Geom",
    "Sam-Gate",
    "Sam-XAttn",
    "Sam-Concat",
    "Sam-Skip",
    "Text-BLIP",
]

# Heatmap panels, each with its own reference and row set:
#   base_min   — gain over Base-min, configs that build directly on the minimal baseline
#   intra_reg  — gain over Intra-Reg, the segment-aware / SAM-fusion / text-encoder family
HEATMAP_PANELS = {
    "basemin": {
        "reference": "Base-min",
        "reference_label": "Base-min",
        "output_stem": "04B_retrieval_heatmap_vs_basemin",
        "rows": [
            "Loss-SigLIP",
            "HN-Syntactic",
            "Aux-ObjCls",
        ],
    },
    "intrareg": {
        "reference": "Intra-Reg",
        "reference_internal_run_ids": ["B0plus"],
        "reference_label": "Intra-Reg",
        "output_stem": "04B_retrieval_heatmap_vs_intrareg",
        "rows": [
            "Seg-Spatial",
            "Seg-Semantic",
            "Seg-Geom",
            "Sam-Gate",
            "Sam-XAttn",
            "Sam-Concat",
            "Sam-Skip",
            "Text-BLIP",
        ],
    },
}

# Heatmap columns: (dataset group, short metric label, source_col, reference_dataset).
# CxC/ECCV/COCO-5K values come from COCO-trained runs; Flickr30K values come from
# Flickr-trained runs.
HEATMAP_RETRIEVAL_COLUMNS_BY_CUTOFF = {
    1: [
        ("Flickr30K", "I2T\nR@1", "summary/test/r1_i2t", "flickr30k"),
        ("Flickr30K", "T2I\nR@1", "summary/test/r1_t2i", "flickr30k"),
        ("COCO 5K", "I2T\nR@1", "summary/test/coco_5k_r1_i2t", "coco"),
        ("COCO 5K", "T2I\nR@1", "summary/test/coco_5k_r1_t2i", "coco"),
        ("CxC", "I2T\nR@1", "summary/test/cxc_r1_i2t", "coco"),
        ("CxC", "T2I\nR@1", "summary/test/cxc_r1_t2i", "coco"),
    ],
    5: [
        ("Flickr30K", "I2T\nR@5", "summary/test/r5_i2t", "flickr30k"),
        ("Flickr30K", "T2I\nR@5", "summary/test/r5_t2i", "flickr30k"),
        ("COCO 5K", "I2T\nR@5", "summary/test/coco_5k_r5_i2t", "coco"),
        ("COCO 5K", "T2I\nR@5", "summary/test/coco_5k_r5_t2i", "coco"),
        ("CxC", "I2T\nR@5", "summary/test/cxc_r5_i2t", "coco"),
        ("CxC", "T2I\nR@5", "summary/test/cxc_r5_t2i", "coco"),
    ],
    10: [
        ("Flickr30K", "I2T\nR@10", "summary/test/r10_i2t", "flickr30k"),
        ("Flickr30K", "T2I\nR@10", "summary/test/r10_t2i", "flickr30k"),
        ("COCO 5K", "I2T\nR@10", "summary/test/coco_5k_r10_i2t", "coco"),
        ("COCO 5K", "T2I\nR@10", "summary/test/coco_5k_r10_t2i", "coco"),
        ("CxC", "I2T\nR@10", "summary/test/cxc_r10_i2t", "coco"),
        ("CxC", "T2I\nR@10", "summary/test/cxc_r10_t2i", "coco"),
    ],
}

HEATMAP_MAPR_COLUMNS = [
    ("ECCV", "I2T\nmAP@R", "summary/test/eccv_map_at_r_i2t", "coco"),
    ("ECCV", "T2I\nmAP@R", "summary/test/eccv_map_at_r_t2i", "coco"),
]

HEATMAP_COLUMN_SETS = {
    "R1": {
        "metric_label": "R@1",
        "output_suffix": "R1",
        "columns": HEATMAP_RETRIEVAL_COLUMNS_BY_CUTOFF[1],
    },
    "R5": {
        "metric_label": "R@5",
        "output_suffix": "R5",
        "columns": HEATMAP_RETRIEVAL_COLUMNS_BY_CUTOFF[5],
    },
    "R10": {
        "metric_label": "R@10",
        "output_suffix": "R10",
        "columns": HEATMAP_RETRIEVAL_COLUMNS_BY_CUTOFF[10],
    },
    "mapr": {
        "metric_label": "mAP@R",
        "output_suffix": "mapr",
        "columns": HEATMAP_MAPR_COLUMNS,
    },
}

HEATMAP_COLUMNS = HEATMAP_COLUMN_SETS["R1"]["columns"]


def _seed_mean(df: pd.DataFrame, source_col: str) -> tuple[float, float, int]:
    """Per-config seed mean/std/count for one source column (already config+dataset scoped).

    Groups by config/seed so duplicate W&B runs for the same seed do not inflate n.
    Returns (mean_pct, std_pct, n_seeds) with NaN/0 when there is no data.
    """
    if source_col not in df.columns:
        return np.nan, np.nan, 0
    values = _numeric_series(df, source_col)
    if "config/seed" in df.columns:
        per_seed = (
            pd.DataFrame({"seed": df["config/seed"].values, "value": values.values})
            .dropna(subset=["value"])
            .groupby("seed", dropna=False)["value"]
            .mean()
        )
        clean = per_seed.dropna()
    else:
        clean = values.dropna()
    if clean.empty:
        return np.nan, np.nan, 0
    pct = clean.map(_to_percent)
    mean = float(pct.mean())
    std = float(pct.std(ddof=1)) if len(pct) >= 2 else np.nan
    return mean, std, int(len(pct))


def _reference_values(
    working: pd.DataFrame,
    reference: str,
    columns: list[tuple[str, str, str, str]],
    reference_internal_run_ids: list[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Per (source_col, dataset) reference value for one panel.

    The reference is a thesis label (Base-min / Intra-Reg) whose per-seed mean
    is the baseline for the deltas.
    """
    if reference_internal_run_ids:
        ref_rows = working[working["internal_run_id"].isin(reference_internal_run_ids)]
    else:
        ref_rows = working[working["thesis_label"].eq(reference)]
    ref: dict[tuple[str, str], float] = {}
    for _group, _label, source_col, ref_ds in columns:
        sub = ref_rows[ref_rows["dataset"].eq(ref_ds)]
        mean, _std, _n = _seed_mean(sub, source_col)
        ref[(source_col, ref_ds)] = mean
    return ref


def build_heatmap_retrieval_data(
    panel: str = "basemin",
    csv_path: str | Path = DEFAULT_CSV_PATH,
    metric_set: str = "R1",
) -> pd.DataFrame:
    """Build one delta-heatmap matrix for a named panel (basemin / intrareg).

    Each cell is the config's per-seed mean minus the panel reference value (zero-shot CLIP,
    Base-min, or Intra-Reg) for that dataset+metric. Rows are taken from the panel spec, in
    order, keeping only those present in the data. Missing / structurally-unavailable cells
    stay NaN (never imputed).
    """
    if panel not in HEATMAP_PANELS:
        raise ValueError(f"Unknown heatmap panel {panel!r}; choose from {list(HEATMAP_PANELS)}")
    if metric_set not in HEATMAP_COLUMN_SETS:
        raise ValueError(f"Unknown heatmap metric set {metric_set!r}; choose from {list(HEATMAP_COLUMN_SETS)}")
    spec = HEATMAP_PANELS[panel]
    column_spec = HEATMAP_COLUMN_SETS[metric_set]
    columns = column_spec["columns"]

    df = load_runs(
        csv_path,
        EXCLUDE,
        result_scope="main",
        drop_superseded=not bool(spec.get("reference_internal_run_ids")),
    )
    if "config/dataset" not in df.columns:
        raise KeyError("Missing required column: config/dataset")
    working = df.copy()
    working["dataset"] = working["config/dataset"].replace({"flickr": "flickr30k"})
    warnings: list[str] = []
    missing_cols = [col for *_h, col, _ds in columns if col not in working.columns]
    if missing_cols:
        warnings.append("Missing source columns: " + ", ".join(sorted(set(missing_cols))))

    ref_vals = _reference_values(working, spec["reference"], columns, spec.get("reference_internal_run_ids"))
    if spec.get("reference_internal_run_ids"):
        ref_ids = ", ".join(spec["reference_internal_run_ids"])
        ref_rows = working[working["internal_run_id"].isin(spec["reference_internal_run_ids"])]
        if ref_rows.empty:
            warnings.append(f"{spec['reference_label']} reference run(s) absent from data: {ref_ids}")
    for (source_col, ref_ds), value in ref_vals.items():
        if not np.isfinite(value):
            warnings.append(f"{spec['reference_label']} reference missing for {source_col} ({ref_ds})")

    present = set(working["thesis_label"])
    row_labels = [lbl for lbl in spec["rows"] if lbl in present]
    dropped = [lbl for lbl in spec["rows"] if lbl not in present]
    if dropped:
        warnings.append("Rows absent from data: " + ", ".join(dropped))

    records = []
    n_cols = len(columns)
    mean = np.full((len(row_labels), n_cols), np.nan)
    delta = np.full_like(mean, np.nan)
    std = np.full_like(mean, np.nan)
    n_seeds = np.zeros_like(mean, dtype=int)
    for r, label in enumerate(row_labels):
        label_rows = working[working["thesis_label"].eq(label)]
        for c, (_group, _metric, source_col, ref_ds) in enumerate(columns):
            sub = label_rows[label_rows["dataset"].eq(ref_ds)]
            m, s, n = _seed_mean(sub, source_col)
            mean[r, c] = m
            std[r, c] = s
            n_seeds[r, c] = n
            ref = ref_vals.get((source_col, ref_ds), np.nan)
            if np.isfinite(m) and np.isfinite(ref):
                delta[r, c] = m - ref
            records.append(
                {
                    "panel": panel,
                    "reference": spec["reference_label"],
                    "thesis_label": label,
                    "dataset_group": _group,
                    "metric": _metric.replace("\n", " "),
                    "source_col": source_col,
                    "reference_dataset": ref_ds,
                    "mean": m,
                    "std": s,
                    "delta": delta[r, c],
                    "n_seeds": n,
                }
            )

    long = pd.DataFrame(records)
    long.attrs["panel"] = panel
    long.attrs["metric_set"] = metric_set
    long.attrs["metric_label"] = column_spec["metric_label"]
    long.attrs["reference_label"] = spec["reference_label"]
    long.attrs["output_stem"] = (
        f"04B_retrieval_heatmap_{column_spec['output_suffix']}_vs_{panel}"
        if metric_set != "mapr"
        else f"04B_retrieval_heatmap_mapr_vs_{panel}"
    )
    long.attrs["alias_output_stem"] = spec["output_stem"] if metric_set == "R1" else None
    long.attrs["row_labels"] = row_labels
    long.attrs["columns"] = columns
    long.attrs["mean"] = mean
    long.attrs["std"] = std
    long.attrs["delta"] = delta
    long.attrs["n_seeds"] = n_seeds
    long.attrs["reference_values"] = {f"{sc}|{ds}": v for (sc, ds), v in ref_vals.items()}
    long.attrs["warnings"] = sorted(set(warnings))
    return long


def print_heatmap_report(data: pd.DataFrame) -> None:
    """Coverage report for the Figure-1 heatmap."""
    row_labels = data.attrs.get("row_labels", [])
    n_seeds = data.attrs.get("n_seeds")
    delta = data.attrs.get("delta")
    columns = data.attrs.get("columns", HEATMAP_COLUMNS)
    metric_label = data.attrs.get("metric_label", "retrieval")
    print(f"Heatmap metric set: {metric_label}")
    print("Heatmap rows:", ", ".join(row_labels) or "(none)")
    print("Heatmap columns:", ", ".join(f"{g} {m}".replace("\n", " ") for g, m, *_ in columns))
    if not data.empty:
        print("Actually plotted (label, dataset, n_seeds):")
        coverage = (
            data[np.isfinite(data["delta"])]
            .groupby(["thesis_label", "reference_dataset"], sort=False)["n_seeds"]
            .agg(lambda values: ",".join(str(int(v)) for v in sorted(set(values))))
            .reset_index()
        )
        for _, row in coverage.iterrows():
            print(f"  ({row['thesis_label']}, {row['reference_dataset']}, {row['n_seeds']})")
    single = []
    missing = []
    for r, label in enumerate(row_labels):
        for c, (group, metric, *_rest) in enumerate(columns):
            tag = f"{label} / {group} {metric}".replace("\n", " ")
            if n_seeds is not None and n_seeds[r, c] == 1:
                single.append(tag)
            if delta is not None and not np.isfinite(delta[r, c]):
                missing.append(tag)
    print(f"Single-seed cells (n=1): {len(single)}")
    for tag in single:
        print(f"  * {tag}")
    print(f"Missing/unavailable cells: {len(missing)}")
    for tag in missing:
        print(f"  -- {tag}")
    warnings = data.attrs.get("warnings", [])
    print("Warnings:")
    for w in warnings:
        print(f"  {w}")
    if not warnings:
        print("  (none)")


def print_retrieval_report(data: pd.DataFrame) -> None:
    print("Detected retrieval configurations:")
    summary = (
        data.groupby(["run_id", "dataset_label"], sort=False)["n_seeds"]
        .max()
        .reset_index()
        .sort_values(["run_id", "dataset_label"])
    )
    for run_id in data.attrs.get("config_order", list(data["run_id"].drop_duplicates())):
        sub = summary[summary["run_id"].eq(run_id)]
        if sub.empty:
            continue
        counts = ", ".join(f"{row['dataset_label']} n={int(row['n_seeds'])}" for _, row in sub.iterrows())
        print(f"  {run_id}: {counts}")

    warnings = data.attrs.get("warnings", [])
    print("Missing data warnings:")
    if warnings:
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("  (none)")


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def format_mean_std(mean: float, std: float, n_seeds: int, precision: int = 1) -> str:
    if pd.isna(mean):
        return "--"
    mean_text = f"{float(mean):.{precision}f}"
    if n_seeds >= 2 and not pd.isna(std):
        return rf"{mean_text} $\pm$ {float(std):.{precision}f}"
    return mean_text + ("*" if n_seeds == 1 else "")


def write_retrieval_table(data: pd.DataFrame, path: Path | None = None) -> None:
    out_path = SAVE_TABLE_DIR / "04_retrieval.tex" if path is None else path
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("i2t", 1),
        ("i2t", 5),
        ("i2t", 10),
        ("t2i", 1),
        ("t2i", 5),
        ("t2i", 10),
    ]
    header = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Run & Dataset & Seeds & R@1 I2T & R@5 I2T & R@10 I2T & R@1 T2I & R@5 T2I & R@10 T2I \\",
        r"\midrule",
    ]
    lines = list(header)
    lookup = data.set_index(["run_id", "dataset", "direction", "k"])
    for run_id in data.attrs.get("config_order", list(data["run_id"].drop_duplicates())):
        for dataset, spec in RETRIEVAL_DATASETS.items():
            sub = data[data["run_id"].eq(run_id) & data["dataset"].eq(dataset)]
            if sub.empty:
                continue
            n_seeds = int(sub["n_seeds"].max())
            latex_label = sub["latex_label"].dropna().iloc[0] if "latex_label" in sub.columns else run_id
            cells = [latex_escape(latex_label), latex_escape(spec["label"]), str(n_seeds)]
            for direction, k in metrics:
                row = lookup.loc[(run_id, dataset, direction, k)]
                cells.append(format_mean_std(row["mean"], row["std"], int(row["n_seeds"])))
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def print_color_report(color_by_config: dict[str, object]) -> None:
    from matplotlib.colors import to_hex

    print("Hex colors by configuration (seaborn colorblind):")
    for run_id, color in color_by_config.items():
        print(f"  {run_id}: {to_hex(color).upper()}")
