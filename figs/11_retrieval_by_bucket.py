from __future__ import annotations

import os
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import FIG_ARTIFACT_ROOT, SAVE_DATA_DIR, SAVE_FIG_DIR, SAVE_TABLE_DIR, latex_escape  # noqa: E402


CACHE_CSV = FIG_ARTIFACT_ROOT / "cache" / "retrieval_by_bucket.csv"
SAVE_DATA_PATH = SAVE_DATA_DIR / "11_retrieval_by_bucket_data.csv"
SAVE_TABLE_PATH = SAVE_TABLE_DIR / "11_retrieval_by_bucket_table.tex"
DATASET_LABELS = {"coco": "COCO", "flickr30k": "Flickr30K"}
DIMENSIONS = {
    "token_quartile": {
        "label": "length",
        "title": "Caption Length Buckets",
        "output": SAVE_FIG_DIR / "11_retrieval_by_bucket_length",
        "xlabel": "Token-count quartile",
    },
    "concept_quartile": {
        "label": "complexity",
        "title": "Caption Complexity Buckets",
        "output": SAVE_FIG_DIR / "11_retrieval_by_bucket_complexity",
        "xlabel": "Concept-count quartile",
    },
}
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]


def configure_matplotlib() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
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


def config_order(run_ids: list[str]) -> list[str]:
    detected = [str(run_id) for run_id in run_ids]
    preferred = ["B0", "B0plus"]
    ordered = [run_id for run_id in preferred if run_id in detected]
    # Match Figure 01 after its two references: simple alphabetical order.
    rest = sorted(run_id for run_id in set(detected) if run_id not in ordered)
    return ordered + rest


def load_data(path: Path = CACHE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing bucketed retrieval CSV: {path}. Run scripts/eval/retrieval_by_bucket.py first.")
    df = pd.read_csv(path)
    required = {"run_id", "seed", "dataset", "direction", "bucket_dim", "bucket_value", "k", "recall", "n_queries"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["run_id", "dataset", "direction", "bucket_dim", "bucket_value", "k"], dropna=False)
    agg = grouped["recall"].agg(mean="mean", std=lambda s: s.std(ddof=1), n_seeds="count").reset_index()
    n_queries = grouped["n_queries"].min().rename("n_queries").reset_index()
    return agg.merge(n_queries, on=["run_id", "dataset", "direction", "bucket_dim", "bucket_value", "k"], how="left")


def y_limits(panel: pd.DataFrame) -> tuple[float, float]:
    lower = panel["mean"] - panel["std"].where(panel["n_seeds"].ge(2), 0.0).fillna(0.0)
    upper = panel["mean"] + panel["std"].where(panel["n_seeds"].ge(2), 0.0).fillna(0.0)
    lo = float(lower.min())
    hi = float(upper.max())
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    return max(0.0, lo - 3.0), min(100.0, hi + 3.0)


def labels_for_configs(data: pd.DataFrame, configs: list[str]) -> dict[str, str]:
    labels = {}
    for run_id in configs:
        n = int(data[data["run_id"].eq(run_id)]["n_seeds"].max())
        labels[run_id] = f"{run_id}{'*' if n == 1 else ''}"
    return labels


def plot_dimension(agg: pd.DataFrame, bucket_dim: str, configs: list[str]) -> None:
    configure_matplotlib()
    spec = DIMENSIONS[bucket_dim]
    data = agg[
        (agg["direction"].eq("t2i"))
        & (agg["bucket_dim"].eq(bucket_dim))
        & (agg["k"].eq(1))
        & (agg["bucket_value"].isin(QUARTILES))
    ].copy()
    if data.empty:
        raise ValueError(f"No R@1 t2i data found for {bucket_dim}.")

    colors = dict(zip(configs, sns.color_palette("colorblind", n_colors=len(configs)), strict=True))
    labels = labels_for_configs(data, configs)
    x = np.arange(len(QUARTILES), dtype=float)
    y_min, y_max = y_limits(data)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.34, wspace=0.12)
    fig.patch.set_facecolor("none")

    for ax, dataset in zip(axes, ["coco", "flickr30k"], strict=True):
        panel = data[data["dataset"].eq(dataset)].copy()
        ax.set_facecolor("none")
        for run_id in configs:
            line = panel[panel["run_id"].eq(run_id)].set_index("bucket_value").reindex(QUARTILES)
            if line["mean"].isna().all():
                continue
            y = line["mean"].to_numpy(dtype=float)
            std = line["std"].to_numpy(dtype=float)
            n_seeds = line["n_seeds"].fillna(0).to_numpy(dtype=int)
            multi = n_seeds >= 2
            linestyle = "-" if multi.any() else "--"
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4.2,
                linewidth=1.25,
                linestyle=linestyle,
                color=colors[run_id],
                label=labels[run_id],
                zorder=3,
            )
            yerr = np.where(multi & np.isfinite(std), std, 0.0)
            if np.any(yerr > 0):
                ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=colors[run_id], elinewidth=0.75, capsize=2.0, alpha=0.75, zorder=2)

        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel(spec["xlabel"])
        ax.set_xticks(x)
        ax.set_xticklabels(QUARTILES)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.7, length=3)
    axes[0].set_ylabel("R@1 (%, zoomed)")

    handles, handle_labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, handle_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        handle_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=min(4, max(1, len(handle_labels))),
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    if any(label.endswith("*") for label in handle_labels):
        fig.text(0.5, 0.025, "* single seed; dashed line", ha="center", va="bottom", fontsize=8, color="0.35")

    output = Path(spec["output"])
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def format_mean_std(mean: float, std: float, n_seeds: int) -> str:
    if pd.isna(mean):
        return "--"
    if n_seeds >= 2 and not pd.isna(std):
        return rf"{mean:.1f} $\pm$ {std:.1f}"
    return f"{mean:.1f}*"


def write_table(agg: pd.DataFrame) -> None:
    data = agg[
        (agg["direction"].eq("t2i"))
        & (agg["k"].eq(1))
        & (agg["bucket_dim"].isin(["token_quartile", "concept_quartile"]))
        & (agg["bucket_value"].isin(QUARTILES))
    ].copy()
    data["bucket_dim_label"] = data["bucket_dim"].map({"token_quartile": "Token", "concept_quartile": "Concept"})
    data = data.sort_values(["dataset", "bucket_dim", "run_id", "bucket_value"])

    lines = [
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Run & Dataset & Bucket type & Bucket & R@1 T2I & Seeds \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        cells = [
            latex_escape(row["run_id"]),
            latex_escape(DATASET_LABELS.get(row["dataset"], row["dataset"])),
            latex_escape(row["bucket_dim_label"]),
            latex_escape(row["bucket_value"]),
            format_mean_std(row["mean"], row["std"], int(row["n_seeds"])),
            str(int(row["n_seeds"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SAVE_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_signal_report(agg: pd.DataFrame) -> None:
    data = agg[(agg["direction"].eq("t2i")) & (agg["k"].eq(1))]
    pivot = data[data["bucket_dim"].isin(["token_quartile", "concept_quartile"])].pivot_table(
        index=["run_id", "dataset", "bucket_dim"],
        columns="bucket_value",
        values="mean",
    )
    flagged = []
    for idx, row in pivot.iterrows():
        if {"Q1", "Q4"}.issubset(row.index) and pd.notna(row["Q1"]) and pd.notna(row["Q4"]):
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
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    agg = aggregate(df)
    configs = config_order(agg["run_id"].dropna().astype(str).unique().tolist())
    agg["config_order"] = agg["run_id"].map({run_id: idx for idx, run_id in enumerate(configs)})
    agg = agg.sort_values(["config_order", "dataset", "direction", "bucket_dim", "bucket_value", "k"]).drop(columns=["config_order"])
    agg.to_csv(SAVE_DATA_PATH, index=False)
    write_table(agg)
    plot_dimension(agg, "token_quartile", configs)
    plot_dimension(agg, "concept_quartile", configs)
    print_signal_report(agg)
    print(f"Wrote {SAVE_DATA_PATH}")
    print(f"Wrote {SAVE_TABLE_PATH}")


if __name__ == "__main__":
    main()
