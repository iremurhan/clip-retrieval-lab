from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/cache/mplconfig")))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import DEFAULT_CSV_PATH, SAVE_DATA_DIR, SAVE_FIG_DIR, SAVE_TABLE_DIR, load_runs, split_by_baseline
from src.eval.mmvp_vlm import PATTERN_ORDER


CACHE_DIR = Path("results/cache/mmvp_vlm")
PATTERN_LABELS = {
    "orientation": "Orientation",
    "presence": "Presence",
    "state": "State",
    "quantity": "Quantity",
    "spatial": "Spatial",
    "color": "Color",
    "structural": "Structural",
    "text_rendering": "Text",
    "viewpoint": "Viewpoint",
    "overall": "Overall",
}


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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_mmvp_cache(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    paths = sorted(cache_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No MMVP-VLM cache JSON files found under {cache_dir}. Run scripts/eval/run_mmvp_local.py first.")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        for pattern in [*PATTERN_ORDER, "overall"]:
            if pattern not in metrics:
                continue
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "dataset": payload.get("dataset", ""),
                    "seed": int(payload.get("seed", -1)),
                    "pattern": pattern,
                    "mean": float(metrics[pattern]) * 100.0,
                }
            )
    return pd.DataFrame(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def config_groups(csv_path: str | Path = DEFAULT_CSV_PATH) -> tuple[list[str], list[str]]:
    runs = load_runs(csv_path)
    b0plus_df, b0_df = split_by_baseline(runs, include_b0_in_interventions=True)
    interventions = list(dict.fromkeys(b0plus_df["config/run_id"].dropna().astype(str)))
    capacity = list(dict.fromkeys(b0_df["config/run_id"].dropna().astype(str)))
    return interventions, capacity


def aggregate(data: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    sub = data[data["run_id"].isin(configs)].copy()
    grouped = sub.groupby(["run_id", "pattern"], sort=False)["mean"]
    out = grouped.agg(["mean", "std", "count"]).reset_index().rename(columns={"count": "n_seeds"})
    out["run_id"] = pd.Categorical(out["run_id"], categories=configs, ordered=True)
    out["pattern"] = pd.Categorical(out["pattern"], categories=[*PATTERN_ORDER, "overall"], ordered=True)
    return out.sort_values(["run_id", "pattern"]).reset_index(drop=True)


def plot_group(data: pd.DataFrame, configs: list[str], output_stem: str, title: str) -> None:
    configure_matplotlib()
    patterns = [*PATTERN_ORDER, "overall"]
    agg = aggregate(data, configs)
    if agg.empty:
        raise ValueError(f"No MMVP-VLM rows available for {title}")
    colors = dict(zip(configs, sns.color_palette("colorblind", n_colors=len(configs)), strict=False))
    x = np.arange(len(patterns), dtype=float)
    width = min(0.82 / max(len(configs), 1), 0.11)
    offsets = (np.arange(len(configs)) - (len(configs) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.29)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    for idx, run_id in enumerate(configs):
        sub = agg[agg["run_id"].astype(str).eq(run_id)].set_index("pattern").reindex(patterns)
        means = sub["mean"].to_numpy(dtype=float)
        stds = sub["std"].to_numpy(dtype=float)
        n_seeds = sub["n_seeds"].fillna(0).to_numpy(dtype=int)
        valid = np.isfinite(means)
        if not valid.any():
            continue
        yerr = np.where((n_seeds >= 2) & np.isfinite(stds), stds, 0.0)
        ax.bar(
            x[valid] + offsets[idx],
            means[valid],
            width=width * 0.90,
            color=colors[run_id],
            edgecolor="white",
            linewidth=0.4,
            yerr=yerr[valid],
            error_kw={"ecolor": "0.15", "elinewidth": 0.65, "capsize": 1.5, "capthick": 0.65},
            zorder=3,
        )

    n_by_run = agg.groupby("run_id", observed=False)["n_seeds"].max().to_dict()
    labels = [f"{run_id}{'*' if int(n_by_run.get(run_id, 0) or 0) == 1 else ''}" for run_id in configs]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[run_id]) for run_id in configs]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=min(5, len(configs)), frameon=False)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS[p] for p in patterns], rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
    ax.tick_params(width=0.7, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.03, "* single seed; dashed line marks chance", ha="center", fontsize=7, color="0.35")

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / f"{output_stem}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{output_stem}.png", dpi=300)
    plt.close(fig)


def write_table(data: pd.DataFrame, configs: list[str], output_name: str) -> None:
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    agg = aggregate(data, configs)
    lookup = agg.set_index(["run_id", "pattern"])
    patterns = [*PATTERN_ORDER, "overall"]
    lines = [
        r"\begin{tabular}{l" + "r" * len(patterns) + "}",
        r"\toprule",
        "Run & " + " & ".join(latex_escape(PATTERN_LABELS[p]) for p in patterns) + r" \\",
        r"\midrule",
    ]
    for run_id in configs:
        if run_id not in set(agg["run_id"].astype(str)):
            continue
        cells = [latex_escape(run_id)]
        for pattern in patterns:
            row = lookup.loc[(run_id, pattern)]
            if pd.isna(row["mean"]):
                cells.append("--")
            elif int(row["n_seeds"]) >= 2 and not pd.isna(row["std"]):
                cells.append(rf"{row['mean']:.1f} $\pm$ {row['std']:.1f}")
            else:
                cells.append(f"{row['mean']:.1f}*")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (SAVE_TABLE_DIR / output_name).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_mmvp_cache()
    interventions, capacity = config_groups()
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(SAVE_DATA_DIR / "09_mmvp_vlm_data.csv", index=False)
    plot_group(data, interventions, "09A_mmvp_vlm_interventions", "MMVP-VLM: interventions")
    plot_group(data, capacity, "09B_mmvp_vlm_capacity", "MMVP-VLM: capacity")
    write_table(data, interventions, "09A_mmvp_vlm_interventions.tex")
    write_table(data, capacity, "09B_mmvp_vlm_capacity.tex")


if __name__ == "__main__":
    main()
