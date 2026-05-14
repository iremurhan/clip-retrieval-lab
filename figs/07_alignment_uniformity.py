from __future__ import annotations

import math
import os
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "cache" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import CACHE_DIR, DEFAULT_CSV_PATH, SAVE_FIG_DIR, SAVE_TABLE_DIR, load_runs

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None


EXCLUDE = ["B0v2", "B0plus_fixed"]
CACHE_CSV = CACHE_DIR / "alignment_uniformity_results.csv"
VALUE_COLS = ["alignment", "uniformity_img", "uniformity_txt", "uniformity_mean"]
GEOMETRY_COLS = {
    "Alignment": "alignment",
    "Uniformity": "uniformity_mean",
}

METRIC_SPECS = [
    ("R@1 I2T", {"flickr30k": "summary/test/r1_i2t", "coco": "summary/test/coco_5k_r1_i2t"}),
    ("R@1 T2I", {"flickr30k": "summary/test/r1_t2i", "coco": "summary/test/coco_5k_r1_t2i"}),
    ("R@5 I2T", {"flickr30k": "summary/test/r5_i2t", "coco": "summary/test/coco_5k_r5_i2t"}),
    ("R@5 T2I", {"flickr30k": "summary/test/r5_t2i", "coco": "summary/test/coco_5k_r5_t2i"}),
    ("ECCV mAP@R I2T", {"coco": "summary/test/eccv_map_at_r_i2t"}),
    ("ECCV mAP@R T2I", {"coco": "summary/test/eccv_map_at_r_t2i"}),
    ("CxC R@1 I2T", {"coco": "summary/test/cxc_r1_i2t"}),
    ("CxC R@1 T2I", {"coco": "summary/test/cxc_r1_t2i"}),
    ("SugarCrepe", {"flickr30k": "sugarcrepe_combined", "coco": "sugarcrepe_combined"}),
]


def configure_matplotlib() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "axes.linewidth": 0.7,
            "axes.edgecolor": "0.2",
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.4f}"
    return rf"{mean:.4f} $\pm$ {std:.4f}"


def fmt_corr(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}"


def load_alignment(path: Path = CACHE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing alignment/uniformity cache: {path}. "
            "Run scripts/eval/run_alignment_uniformity_local.py first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(
            f"Alignment/uniformity cache is empty: {path}. "
            "Run scripts/eval/run_alignment_uniformity_local.py without --dry-run first."
        )
    df = df[~df["run_id"].isin(EXCLUDE)].copy()
    df["dataset"] = df["dataset"].replace({"flickr": "flickr30k"})
    for col in ["seed", *VALUE_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[VALUE_COLS].isna().any().any():
        missing = df[df[VALUE_COLS].isna().any(axis=1)]
        raise ValueError(f"Alignment/uniformity cache has missing values:\n{missing}")
    return df


def aggregate_alignment(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["run_id", "dataset"], dropna=False)
    mean_df = grouped[VALUE_COLS].mean().add_suffix("_mean")
    std_df = grouped[VALUE_COLS].std(ddof=1).add_suffix("_std")
    n_df = grouped.size().rename("n_seeds")
    return pd.concat([mean_df, std_df, n_df], axis=1).reset_index().sort_values(["dataset", "run_id"])


def write_alignment_table(agg: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Run & Dataset & Alignment & Uniformity$_\mathrm{img}$ & Uniformity$_\mathrm{txt}$ & Uniformity$_\mathrm{mean}$ \\",
        r"\midrule",
    ]
    for _, row in agg.iterrows():
        cells = [
            latex_escape(row["run_id"]),
            latex_escape(row["dataset"]),
            fmt_mean_std(row["alignment_mean"], row["alignment_std"]),
            fmt_mean_std(row["uniformity_img_mean"], row["uniformity_img_std"]),
            fmt_mean_std(row["uniformity_txt_mean"], row["uniformity_txt_std"]),
            fmt_mean_std(row["uniformity_mean_mean"], row["uniformity_mean_std"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_TABLE_DIR / "07_alignment_uniformity_table.tex").write_text("\n".join(lines), encoding="utf-8")


def build_correlation_data(alignment: pd.DataFrame, runs_csv: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    runs = load_runs(runs_csv, EXCLUDE).copy()
    runs["dataset"] = runs["config/dataset"].replace({"flickr": "flickr30k"})
    runs["run_id"] = runs["config/run_id"]
    runs["seed"] = pd.to_numeric(runs["config/seed"], errors="coerce")
    for col in ["summary/sugarcrepe/macro_avg", "summary/sugarcrepe/overall"]:
        if col not in runs:
            runs[col] = np.nan
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
    runs["sugarcrepe_combined"] = runs["summary/sugarcrepe/macro_avg"].combine_first(
        runs["summary/sugarcrepe/overall"]
    )

    needed_cols = {
        "run_id",
        "dataset",
        "seed",
        "sugarcrepe_combined",
    }
    for _name, per_dataset in METRIC_SPECS:
        needed_cols.update(per_dataset.values())
    for col in needed_cols:
        if col not in runs:
            runs[col] = np.nan

    merged = alignment.merge(
        runs[list(needed_cols)],
        on=["run_id", "dataset", "seed"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No rows matched between alignment cache and runs_summary.csv.")
    return merged


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, np.nan
    if stats is None:
        return valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"), np.nan, valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")
    pearson = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    spearman = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic)


def compute_correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_label, per_dataset in METRIC_SPECS:
        for geom_label, geom_col in GEOMETRY_COLS.items():
            row = {"metric": metric_label, "quantity": geom_label}
            for dataset in ["flickr30k", "coco"]:
                metric_col = per_dataset.get(dataset)
                subset = data[data["dataset"] == dataset]
                if metric_col is None or metric_col not in subset:
                    r, p, rho = np.nan, np.nan, np.nan
                else:
                    metric_values = pd.to_numeric(subset[metric_col], errors="coerce")
                    r, p, rho = corr_pair(subset[geom_col], metric_values)
                row[f"{dataset}_pearson"] = r
                row[f"{dataset}_p"] = p
                row[f"{dataset}_spearman"] = rho
            rows.append(row)
    return pd.DataFrame(rows)


def write_correlation_table(corrs: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Metric & Quantity & Flickr $r$ & Flickr $p$ & Flickr $\rho$ & COCO $r$ & COCO $p$ & COCO $\rho$ \\",
        r"\midrule",
    ]
    for _, row in corrs.iterrows():
        cells = [
            latex_escape(row["metric"]),
            latex_escape(row["quantity"]),
            fmt_corr(row["flickr30k_pearson"]),
            fmt_p(row["flickr30k_p"]),
            fmt_corr(row["flickr30k_spearman"]),
            fmt_corr(row["coco_pearson"]),
            fmt_p(row["coco_p"]),
            fmt_corr(row["coco_spearman"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_TABLE_DIR / "07_alignment_uniformity_correlations.tex").write_text("\n".join(lines), encoding="utf-8")


def family_for_run(run_id: str) -> str:
    if run_id == "BLIP_TEXT" or run_id.startswith("BLIP_TEXT"):
        return "text-encoder"
    if run_id.startswith("B5"):
        return "segment-aware"
    if run_id.startswith("B4"):
        return "aux-cls"
    if run_id.startswith("B2"):
        return "hard-neg"
    if run_id.startswith("B1"):
        return "loss"
    if run_id.startswith("B0"):
        return "baseline/capacity"
    return "other"


def axis_limits(values: pd.Series, pad_frac: float = 0.14) -> tuple[float, float]:
    lo = float(values.min())
    hi = float(values.max())
    pad = max((hi - lo) * pad_frac, 1e-4)
    return lo - pad, hi + pad


def plot_scatter(agg: pd.DataFrame) -> None:
    configure_matplotlib()
    data = agg.copy()
    data["family"] = data["run_id"].map(family_for_run)

    dataset_palette = dict(zip(["coco", "flickr30k"], sns.color_palette("colorblind", 2), strict=True))
    markers = {
        "baseline/capacity": "o",
        "loss": "s",
        "hard-neg": "D",
        "aux-cls": "^",
        "segment-aware": "P",
        "text-encoder": "X",
        "other": "v",
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.15))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.25)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    for _, row in data.iterrows():
        color = dataset_palette.get(row["dataset"], "0.35")
        marker = markers.get(row["family"], "o")
        single_seed = int(row["n_seeds"]) == 1
        x = row["alignment_mean"]
        y = row["uniformity_mean_mean"]

        if single_seed:
            ax.scatter(
                x,
                y,
                marker=marker,
                s=42,
                facecolors="white",
                edgecolors=[color],
                linewidths=1.1,
                zorder=4,
            )
        else:
            ax.errorbar(
                x,
                y,
                xerr=row["alignment_std"],
                yerr=row["uniformity_mean_std"],
                fmt=marker,
                markersize=5.5,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                ecolor="0.2",
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                zorder=4,
            )

    texts = []
    offsets = [(4, 4), (6, -7), (-8, 4), (-10, -7), (7, 10), (-12, 10), (10, -12), (-14, -12)]
    for idx, row in data.reset_index(drop=True).iterrows():
        dx, dy = offsets[idx % len(offsets)]
        label = f"{row['run_id']}{'*' if int(row['n_seeds']) == 1 else ''}"
        texts.append(
            ax.annotate(
                label,
                xy=(row["alignment_mean"], row["uniformity_mean_mean"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=5.4,
                color="0.18",
                zorder=5,
            )
        )
    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.15, 1.25),
            expand_text=(1.05, 1.15),
            arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.35},
        )

    ax.set_xlim(*axis_limits(data["alignment_mean"]))
    ax.set_ylim(*axis_limits(data["uniformity_mean_mean"]))
    ax.set_xlabel("Alignment (lower is better)")
    ax.set_ylabel("Mean uniformity (lower is better)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)
    ax.grid(axis="both", color="0.9", linewidth=0.5, zorder=0)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, markersize=5, label=dataset)
        for dataset, color in dataset_palette.items()
        if dataset in set(data["dataset"])
    ]
    family_handles = [
        plt.Line2D([0], [0], marker=marker, color="0.25", markerfacecolor="white", markeredgecolor="0.25", linestyle="None", markersize=5, label=family)
        for family, marker in markers.items()
        if family in set(data["family"])
    ]
    ax.legend(
        handles=dataset_handles + family_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    fig.text(0.5, 0.02, "* single seed", ha="center", va="bottom", color="0.35", fontsize=7)

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / "07_alignment_uniformity_scatter.pdf")
    fig.savefig(SAVE_FIG_DIR / "07_alignment_uniformity_scatter.png", dpi=300)
    plt.close(fig)


def print_summary(alignment: pd.DataFrame, corrs: pd.DataFrame) -> None:
    align_min = alignment["alignment"].min()
    align_max = alignment["alignment"].max()
    unif_min = alignment["uniformity_mean"].min()
    unif_max = alignment["uniformity_mean"].max()

    long_rows = []
    for _, row in corrs.iterrows():
        for dataset in ["flickr30k", "coco"]:
            r = row[f"{dataset}_pearson"]
            if not pd.isna(r):
                long_rows.append((abs(r), r, dataset, row["metric"], row["quantity"], row[f"{dataset}_p"]))
    strongest = max(long_rows, default=None, key=lambda item: item[0])

    print(f"Alignment range: {align_min:.4f} to {align_max:.4f}")
    print(f"Uniformity_mean range: {unif_min:.4f} to {unif_max:.4f}")
    if strongest is not None:
        _, r, dataset, metric, quantity, p = strongest
        print(
            "Strongest Pearson correlation: "
            f"{dataset} {metric} vs {quantity}, r={r:.3f}, p={fmt_p(p)}"
        )


def main() -> None:
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    alignment = load_alignment()
    agg = aggregate_alignment(alignment)
    write_alignment_table(agg)

    corr_data = build_correlation_data(alignment)
    corrs = compute_correlations(corr_data)
    write_correlation_table(corrs)
    plot_scatter(agg)
    print_summary(alignment, corrs)


if __name__ == "__main__":
    main()
