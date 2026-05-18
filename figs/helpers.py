from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
FIG_ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_FIG_ARTIFACT_ROOT", ARTIFACT_ROOT / "figs")
)
SAVE_FIG_DIR = Path(os.environ.get("SAVE_FIG_DIR", FIG_ARTIFACT_ROOT / "figures"))
SAVE_TABLE_DIR = Path(os.environ.get("SAVE_TABLE_DIR", FIG_ARTIFACT_ROOT / "tables"))
SAVE_DATA_DIR = Path(os.environ.get("SAVE_DATA_DIR", FIG_ARTIFACT_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ARTIFACT_ROOT / "cache"))
DEFAULT_CSV_PATH = Path("/Volumes/T7/Research/wandb/runs_summary.csv")
EXCLUDE = ["B0v2", "B0plus_fixed", "B5_seg"]


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

PANEL_LAYOUT = [
    ("Add", ["add_att", "add_obj"], (85, 99)),
    ("Replace", ["replace_att", "replace_obj", "replace_rel"], (78, 99)),
    ("Swap", ["swap_att", "swap_obj"], (60, 80)),
]

RETRIEVAL_CONFIG_ORDER = [
    "B0",
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
    "BLIP_TEXT",
]

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


def load_runs(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    exclude_list: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load CSV, apply EXCLUDE filter, return DataFrame."""
    df = pd.read_csv(csv_path)
    if "config/run_id" not in df.columns:
        raise KeyError("Expected column 'config/run_id' in runs summary CSV.")

    df = df[df["config/run_id"].notna()].copy()
    excludes = EXCLUDE if exclude_list is None else list(exclude_list)
    excluded = _string_values(set(df["config/run_id"]) & set(excludes))
    filtered = df[~df["config/run_id"].isin(excludes)].copy()
    filtered.attrs["excluded_hard"] = excluded
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
        raise KeyError(f"Missing required columns: {missing}")

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
    """Group by config/run_id, return DataFrame with mean, std, n_seeds per value_col."""
    if df.empty:
        columns = ["config/run_id", "run_id", "n_seeds"]
        for col in value_cols:
            columns.extend([f"{col}_mean", f"{col}_std"])
        return pd.DataFrame(columns=columns)

    missing = [col for col in value_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing value columns: {missing}")

    grouped = df.groupby("config/run_id", sort=False, dropna=False)
    mean_df = grouped[value_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped[value_cols].std(ddof=1, numeric_only=True).add_suffix("_std")
    n_df = grouped.size().rename("n_seeds")
    out = pd.concat([mean_df, std_df, n_df], axis=1).reset_index()
    out.insert(1, "run_id", out["config/run_id"])
    return out


def aggregate_seeds(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Backward-compatible aggregation used by older figure scripts."""
    grouped = df.groupby(["config/run_id", "config/dataset"], dropna=False)
    mean_df = grouped[value_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped[value_cols].std(ddof=1, numeric_only=True).add_suffix("_std")
    n_df = grouped.size().rename("n_seeds")
    return pd.concat([mean_df, std_df, n_df], axis=1).reset_index()


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


def _to_percent(value: object) -> float:
    if pd.isna(value):
        return np.nan
    value = float(value)
    return value * 100.0 if abs(value) <= 1.5 else value


def build_sugarcrepe_data(df: pd.DataFrame, config_order: list[str]) -> pd.DataFrame:
    value_cols = [f"sc_{category}" for category, _ in SC_CATEGORIES] + ["sc_overall"]
    aggregate = aggregate_by_config(df, value_cols)
    return sugarcrepe_aggregate_to_long(aggregate, config_order)


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
                    "category": category,
                    "mean": _to_percent(run[f"{value_col}_mean"]),
                    "std": _to_percent(run[f"{value_col}_std"]),
                    "n_seeds": int(run["n_seeds"]),
                }
            )
    return pd.DataFrame(rows, columns=["run_id", "category", "mean", "std", "n_seeds"])


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
    config_order = retrieval_config_order(working.loc[has_any_retrieval, "config/run_id"].unique())
    no_metric_runs = sorted(set(working["config/run_id"].dropna().astype(str)) - set(config_order))
    if no_metric_runs:
        warnings.append("Rows with no primary retrieval metrics excluded: " + ", ".join(no_metric_runs))

    suggested = set(RETRIEVAL_CONFIG_ORDER)
    extras = [run_id for run_id in config_order if run_id not in suggested]
    if extras:
        warnings.append("Additional configs appended after suggested order: " + ", ".join(extras))

    rows = []
    for dataset, spec in RETRIEVAL_DATASETS.items():
        dataset_df = working[working["dataset"].eq(dataset)].copy()
        present = set(dataset_df["config/run_id"].dropna().astype(str))
        for run_id in config_order:
            if run_id not in present:
                warnings.append(f"{spec['label']}: no rows for {run_id}")
                continue

            run_df = dataset_df[dataset_df["config/run_id"].eq(run_id)]
            dataset_cols = list(spec["columns"].values())
            has_dataset_metrics = run_df[dataset_cols].apply(lambda col: pd.to_numeric(col, errors="coerce")).notna().to_numpy().any()
            if not has_dataset_metrics:
                warnings.append(f"{spec['label']} {run_id}: no primary retrieval metrics")
                continue

            for direction in RETRIEVAL_DIRECTIONS:
                for k in RETRIEVAL_KS:
                    source_col = spec["columns"][(direction, k)]
                    values = _numeric_series(run_df, source_col).dropna().map(_to_percent)
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
            cells = [latex_escape(run_id), latex_escape(spec["label"]), str(n_seeds)]
            for direction, k in metrics:
                row = lookup.loc[(run_id, dataset, direction, k)]
                cells.append(format_mean_std(row["mean"], row["std"], int(row["n_seeds"])))
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def print_std_report(data: pd.DataFrame) -> None:
    category_order = [category for category, _ in SC_CATEGORIES] + ["overall"]
    print("Std values by config x category (% points):")
    for run_id, sub in data.groupby("run_id", sort=False):
        n_seeds = int(sub["n_seeds"].max())
        print(f"  {run_id} (n_seeds={n_seeds}):")
        ordered = sub.set_index("category").reindex(category_order)
        for category, row in ordered.iterrows():
            std = row["std"]
            std_text = "NA" if pd.isna(std) else f"{float(std):.4f}"
            print(f"    {category}: std={std_text}")


def print_color_report(color_by_config: dict[str, object]) -> None:
    from matplotlib.colors import to_hex

    print("Hex colors by configuration (seaborn colorblind):")
    for run_id, color in color_by_config.items():
        print(f"  {run_id}: {to_hex(color).upper()}")


def _save_deuteranopia_check(source_png: Path, check_png: Path) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    image = mpimg.imread(source_png)
    rgb = image[..., :3].astype("float64", copy=False)
    rgb = np.nan_to_num(rgb, nan=1.0, posinf=1.0, neginf=0.0)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    alpha = image[..., 3:] if image.shape[-1] == 4 else None
    red = 0.625 * rgb[..., 0] + 0.375 * rgb[..., 1]
    green = 0.700 * rgb[..., 0] + 0.300 * rgb[..., 1]
    blue = 0.300 * rgb[..., 1] + 0.700 * rgb[..., 2]
    simulated = np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)
    if alpha is not None:
        simulated = np.concatenate([simulated, alpha], axis=-1)
    plt.imsave(check_png, simulated)


def plot_sugarcrepe_panels(
    data: pd.DataFrame,
    output_stem: str,
    save_colorblind_check: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(data["run_id"].drop_duplicates())
    if not configs:
        raise ValueError("No configurations available to plot.")

    palette = get_palette(len(configs))
    color_by_config = dict(zip(configs, palette))
    print_color_report(color_by_config)
    print_std_report(data)

    n_by_config = data.groupby("run_id", sort=False)["n_seeds"].max().to_dict()
    labels = {run_id: f"{run_id}{'*' if int(n_by_config[run_id]) == 1 else ''}" for run_id in configs}
    error_kw = {"capsize": 3, "capthick": 1, "ecolor": "black", "elinewidth": 1}

    fig = plt.figure(figsize=(8.5, 4.5))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[2, 3, 2],
        height_ratios=[4.5, 1.5],
        left=0.075,
        right=0.99,
        top=0.91,
        bottom=0.30,
        wspace=0.16,
        hspace=0.48,
    )
    panel_axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    overall_ax = fig.add_subplot(gs[1, :])

    for panel_idx, (ax, (title, categories, y_range)) in enumerate(zip(panel_axes, PANEL_LAYOUT)):
        x = np.arange(len(categories), dtype=float)
        width = min(0.82 / len(configs), 0.14)
        offsets = (np.arange(len(configs)) - (len(configs) - 1) / 2.0) * width

        for config_idx, run_id in enumerate(configs):
            sub = data[(data["run_id"] == run_id) & (data["category"].isin(categories))]
            sub = sub.set_index("category").reindex(categories)
            means = sub["mean"].to_numpy(dtype=float)
            stds = sub["std"].to_numpy(dtype=float)
            xpos = x + offsets[config_idx]
            yerr = np.nan_to_num(stds, nan=0.0) if int(n_by_config[run_id]) >= 2 else None
            ax.bar(
                xpos,
                means,
                width=width * 0.92,
                color=color_by_config[run_id],
                edgecolor="white",
                linewidth=0.55,
                yerr=yerr,
                error_kw=error_kw,
                zorder=3,
            )

        label_map = dict(SC_CATEGORIES)
        ax.set_title(title, pad=5, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[category] for category in categories])
        ax.set_ylim(*y_range)
        ax.set_xlim(-0.55, len(categories) - 0.45)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(axis="y", which="minor", color="0.82", linewidth=0.45, alpha=0.3, zorder=0)
        ax.grid(axis="y", which="major", color="0.84", linewidth=0.5, alpha=0.3, zorder=0)
        ax.tick_params(width=0.7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_idx == 0:
            ax.set_ylabel("Accuracy (%)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    overall = data[data["category"] == "overall"].set_index("run_id").reindex(configs)
    x = np.arange(len(configs), dtype=float)
    means = overall["mean"].to_numpy(dtype=float)
    stds = overall["std"].to_numpy(dtype=float)
    for idx, run_id in enumerate(configs):
        yerr = np.array([stds[idx]]) if int(n_by_config[run_id]) >= 2 and np.isfinite(stds[idx]) else None
        overall_ax.bar(
            x[idx],
            means[idx],
            width=0.72,
            color=color_by_config[run_id],
            edgecolor="white",
            linewidth=0.55,
            yerr=yerr,
            error_kw=error_kw,
            zorder=3,
        )
    finite_means = means[np.isfinite(means)]
    if finite_means.size:
        ymin = float(finite_means.min() - 1.0)
        ymax = float(finite_means.max() + 1.0)
        if np.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0
        overall_ax.set_ylim(ymin, ymax)
    overall_ax.set_ylabel("Overall (%)")
    overall_ax.set_xticks(x)
    overall_ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
    overall_ax.grid(axis="y", which="major", color="0.84", linewidth=0.5, alpha=0.3, zorder=0)
    overall_ax.tick_params(width=0.7, length=3)
    overall_ax.spines["top"].set_visible(False)
    overall_ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_by_config[run_id]) for run_id in configs]
    fig.legend(
        handles,
        [labels[run_id] for run_id in configs],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=len(configs) if len(configs) <= 6 else 5,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
    )
    fig.text(0.5, 0.02, "* single seed", ha="center", va="center", fontsize=8, color="0.35")

    fig.savefig(SAVE_FIG_DIR / f"{output_stem}.pdf")
    png_path = SAVE_FIG_DIR / f"{output_stem}.png"
    fig.savefig(png_path, dpi=300)
    if save_colorblind_check:
        _save_deuteranopia_check(png_path, SAVE_FIG_DIR / f"{output_stem}_colorblind_check.png")
    plt.close(fig)


def print_family_report(df: pd.DataFrame, b0plus_df: pd.DataFrame, b0_df: pd.DataFrame) -> None:
    def print_configs(title: str, family_df: pd.DataFrame) -> None:
        counts = family_df.groupby("config/run_id", sort=True).size()
        print(title)
        if counts.empty:
            print("  (none)")
            return
        for run_id, n_seeds in counts.items():
            print(f"  {run_id}: n_seeds={int(n_seeds)}")

    print_configs("Detected B0plus family configurations:", b0plus_df)
    print_configs("Detected B0 family configurations:", b0_df)

    print("Excluded due to filters:")
    excluded_any = False
    for label, key in [
        ("  hard run_id filter", "excluded_hard"),
        ("  non-COCO rows with SugarCrepe", "excluded_non_coco_with_sugarcrepe"),
        ("  COCO rows without SugarCrepe", "excluded_coco_without_sugarcrepe"),
        ("  unmatched family weights", "excluded_no_family"),
    ]:
        values = df.attrs.get(key, []) or b0plus_df.attrs.get(key, []) or b0_df.attrs.get(key, [])
        if values:
            excluded_any = True
            print(f"{label}: {', '.join(values)}")
    if not excluded_any:
        print("  (none)")
