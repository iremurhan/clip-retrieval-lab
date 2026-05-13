from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/cache/mplconfig")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import CACHE_DIR, SAVE_FIG_DIR, SAVE_TABLE_DIR


EXCLUDE = ["B0v2", "B0plus_fixed"]
CACHE_CSV = CACHE_DIR / "patch_level_results.csv"
CATEGORIES = [
    ("add_att", "Add\nAtt"),
    ("add_obj", "Add\nObj"),
    ("replace_att", "Replace\nAtt"),
    ("replace_obj", "Replace\nObj"),
    ("replace_rel", "Replace\nRel"),
    ("swap_att", "Swap\nAtt"),
    ("swap_obj", "Swap\nObj"),
    ("macro_avg", "Macro"),
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
            "axes.titlesize": 8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def latex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:+.3f}"
    return rf"{mean:+.3f} $\pm$ {std:.3f}"


def config_label(run_id: str, dataset: str) -> str:
    return f"{run_id} ({dataset})"


def load_results(path: Path = CACHE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing patch-level cache: {path}. Run scripts/eval/run_patch_level_local.py first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Patch-level cache is empty: {path}")
    df = df[~df["run_id"].isin(EXCLUDE)].copy()
    df["dataset"] = df["dataset"].replace({"flickr": "flickr30k"})
    for col in ["seed", "cls_accuracy", "patch_max_accuracy", "delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["cls_accuracy", "patch_max_accuracy", "delta"]].isna().any().any():
        missing = df[df[["cls_accuracy", "patch_max_accuracy", "delta"]].isna().any(axis=1)]
        raise ValueError(f"Patch-level cache has missing values:\n{missing}")
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["run_id", "dataset", "subcategory"], dropna=False)
    mean_df = grouped[["cls_accuracy", "patch_max_accuracy", "delta"]].mean().add_suffix("_mean")
    std_df = grouped[["cls_accuracy", "patch_max_accuracy", "delta"]].std(ddof=1).add_suffix("_std")
    n_df = grouped.size().rename("n_seeds")
    agg = pd.concat([mean_df, std_df, n_df], axis=1).reset_index()
    agg["config"] = [config_label(r, d) for r, d in zip(agg["run_id"], agg["dataset"], strict=True)]
    return agg


def ordered_configs(agg: pd.DataFrame) -> list[str]:
    order = (
        agg[["config", "run_id", "dataset"]]
        .drop_duplicates()
        .sort_values(["dataset", "run_id"])
    )
    return order["config"].tolist()


def plot_grouped_bars(agg: pd.DataFrame) -> None:
    configure_matplotlib()
    configs = ordered_configs(agg)
    n_configs = len(configs)
    ncols = 4
    nrows = max(1, math.ceil(n_configs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, max(2.2, 1.55 * nrows)), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.12, wspace=0.16, hspace=0.55)
    fig.patch.set_facecolor("none")

    x = np.arange(len(CATEGORIES), dtype=float)
    labels = [label for _, label in CATEGORIES]
    cls_color = sns.color_palette("colorblind", 2)[0]
    patch_color = sns.color_palette("colorblind", 2)[1]

    for ax, config in zip(axes, configs, strict=False):
        sub = agg[agg["config"] == config].set_index("subcategory").reindex([name for name, _ in CATEGORIES])
        cls = sub["cls_accuracy_mean"].to_numpy() * 100.0
        patch = sub["patch_max_accuracy_mean"].to_numpy() * 100.0
        cls_std = sub["cls_accuracy_std"].to_numpy() * 100.0
        patch_std = sub["patch_max_accuracy_std"].to_numpy() * 100.0
        n_seed = int(sub["n_seeds"].max())
        width = 0.36

        ax.set_facecolor("none")
        ax.bar(x - width / 2, cls, width=width, color=cls_color, alpha=0.45, edgecolor="white", linewidth=0.5, label="CLS", zorder=3)
        ax.bar(x + width / 2, patch, width=width, color=patch_color, alpha=0.85, edgecolor="white", linewidth=0.5, label="Patch-max", zorder=3)
        if n_seed > 1:
            ax.errorbar(x - width / 2, cls, yerr=cls_std, fmt="none", ecolor="0.15", elinewidth=0.55, capsize=1.2, zorder=4)
            ax.errorbar(x + width / 2, patch, yerr=patch_std, fmt="none", ecolor="0.15", elinewidth=0.55, capsize=1.2, zorder=4)

        ax.axvline(6.5, color="0.78", linewidth=0.6, zorder=2)
        ax.set_title(f"{config}{'*' if n_seed == 1 else ''}", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(45, 100)
        ax.grid(axis="y", color="0.9", linewidth=0.45, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.6, length=2.5)

    for ax in axes[len(configs) :]:
        ax.axis("off")

    axes[0].set_ylabel("Accuracy (%)")
    handles = [
        plt.Line2D([0], [0], color=cls_color, alpha=0.45, linewidth=6, label="CLS"),
        plt.Line2D([0], [0], color=patch_color, alpha=0.85, linewidth=6, label="Patch-max"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False)
    fig.text(0.5, 0.005, "* single seed", ha="center", va="bottom", color="0.35", fontsize=7)

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / "08_patch_vs_cls.pdf")
    fig.savefig(SAVE_FIG_DIR / "08_patch_vs_cls.png", dpi=300)
    plt.close(fig)


def plot_delta_heatmap(agg: pd.DataFrame) -> None:
    configure_matplotlib()
    configs = ordered_configs(agg)
    category_order = [name for name, _ in CATEGORIES]
    category_labels = [label.replace("\n", " ") for _, label in CATEGORIES]
    matrix = (
        agg.pivot_table(index="config", columns="subcategory", values="delta_mean")
        .reindex(index=configs, columns=category_order)
        * 100.0
    )
    max_abs = float(np.nanmax(np.abs(matrix.to_numpy())))
    vmax = max(max_abs, 0.5)

    height = max(2.8, 0.28 * len(configs) + 1.1)
    fig, ax = plt.subplots(figsize=(6.5, height))
    fig.subplots_adjust(left=0.25, right=0.98, top=0.94, bottom=0.16)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Patch-max minus CLS (pp)"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(category_labels, rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / "08_patch_vs_cls_delta_heatmap.pdf")
    fig.savefig(SAVE_FIG_DIR / "08_patch_vs_cls_delta_heatmap.png", dpi=300)
    plt.close(fig)


def write_delta_table(agg: pd.DataFrame) -> None:
    category_order = [name for name, _ in CATEGORIES]
    category_headers = [label.replace("\n", " ") for _, label in CATEGORIES]
    configs = ordered_configs(agg)
    lookup = agg.set_index(["config", "subcategory"])

    col_spec = "l" + "r" * len(category_order)
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Configuration & " + " & ".join(latex_escape(h) for h in category_headers) + r" \\",
        r"\midrule",
    ]
    for config in configs:
        cells = [latex_escape(config)]
        for category in category_order:
            row = lookup.loc[(config, category)]
            cells.append(fmt_mean_std(row["delta_mean"] * 100.0, row["delta_std"] * 100.0))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_TABLE_DIR / "08_patch_vs_cls_table.tex").write_text("\n".join(lines), encoding="utf-8")


def print_summary(agg: pd.DataFrame) -> None:
    no_macro = agg[agg["subcategory"] != "macro_avg"].copy()
    strongest = no_macro.reindex(no_macro["delta_mean"].abs().sort_values(ascending=False).index).head(5)
    swap = no_macro[no_macro["subcategory"].isin(["swap_att", "swap_obj"])]
    non_swap = no_macro[~no_macro["subcategory"].isin(["swap_att", "swap_obj"])]
    swap_abs = swap["delta_mean"].abs().mean() * 100.0
    non_swap_abs = non_swap["delta_mean"].abs().mean() * 100.0

    print("Largest absolute patch-vs-CLS deltas:")
    for _, row in strongest.iterrows():
        print(f"  {row['config']} {row['subcategory']}: {row['delta_mean'] * 100.0:+.2f} pp")
    print(f"Mean absolute swap delta: {swap_abs:.2f} pp")
    print(f"Mean absolute non-swap delta: {non_swap_abs:.2f} pp")
    if swap_abs <= non_swap_abs:
        print("Swap categories are not the largest average patch-vs-CLS signal in this cache.")


def main() -> None:
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    agg = aggregate_results(data)
    plot_grouped_bars(agg)
    plot_delta_heatmap(agg)
    write_delta_table(agg)
    print_summary(agg)


if __name__ == "__main__":
    main()
