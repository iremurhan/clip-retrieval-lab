#!/usr/bin/env python3
"""
SugarCrepe–Retrieval Correspondence Analysis
=============================================
Quantifies whether retrieval improvements correspond to compositional-
discrimination improvements across the intervention ladder.

Outputs (saved under analysis/sugarcrepe_correspondence/<timestamp>/):
  delta_table.csv          — long-form table of absolute values and deltas
  correlation_analysis.txt — Pearson / Spearman / Kendall with bootstrap CIs
  subcategory_heatmap.png  — Spearman heatmap: Δ R@1 T2I vs Δ SC subcategory
  quadrant_scatter.png     — scatter per dataset (x=ΔR@1T2I, y=ΔSC overall)
  summary.md               — ~300-word narrative summary
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — correlation analysis will be skipped.", file=sys.stderr)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

CSV_PATH = Path("/Volumes/T7/Research/wandb/runs_summary.csv")

# Registry parent assignments (from configs/registry.yaml)
# Runs in REFERENCE_RUNS supply the baseline; all others are interventions.
REFERENCE_RUNS = {"B0", "B0plus_fixed"}

REFERENCE_MAP: dict[str, str] = {
    # B0 family (parent: B0 in registry)
    "B0plus":                    "B0",   # B0+ adds intra-modal contrast on top of B0
    "B0_proj1024":               "B0",
    "B0_projonly":               "B0",
    "B0_uf5":                    "B0",
    "B0_uf6":                    "B0",
    "B0_uf7":                    "B0",
    "B1":                        "B0",
    "B2":                        "B0",
    "B4":                        "B0",
    # B0plus_fixed family (parent: B0plus_fixed in registry)
    "B5a_seg_spatial":           "B0plus_fixed",
    "B5b_seg_semantic":          "B0plus_fixed",
    "B5c_seg_continuous":        "B0plus_fixed",
    "B5d_multistream_gate":      "B0plus_fixed",
    "B5d_multistream_crossattn": "B0plus_fixed",
    "B5d_multistream_concat":    "B0plus_fixed",
    "B5e_sam_skip":              "B0plus_fixed",
    "BLIP_TEXT":                 "B0plus_fixed",
}

SC_CATEGORIES = [
    "add_att", "add_obj",
    "replace_att", "replace_obj", "replace_rel",
    "swap_att", "swap_obj",
]
SC_CATEGORY_LABELS = {
    "add_att":      "Add Attr",
    "add_obj":      "Add Obj",
    "replace_att":  "Repl Attr",
    "replace_obj":  "Repl Obj",
    "replace_rel":  "Repl Rel",
    "swap_att":     "Swap Attr",
    "swap_obj":     "Swap Obj",
}

# Retrieval column specs per (dataset_key, direction, k) → W&B column name
RETRIEVAL_SPECS: dict[str, dict[tuple[str, int], str]] = {
    "flickr30k": {
        ("i2t", 1):  "summary/test/r1_i2t",
        ("i2t", 5):  "summary/test/r5_i2t",
        ("i2t", 10): "summary/test/r10_i2t",
        ("t2i", 1):  "summary/test/r1_t2i",
        ("t2i", 5):  "summary/test/r5_t2i",
        ("t2i", 10): "summary/test/r10_t2i",
    },
    "coco": {
        ("i2t", 1):  "summary/test/coco_5k_r1_i2t",
        ("i2t", 5):  "summary/test/coco_5k_r5_i2t",
        ("i2t", 10): "summary/test/coco_5k_r10_i2t",
        ("t2i", 1):  "summary/test/coco_5k_r1_t2i",
        ("t2i", 5):  "summary/test/coco_5k_r5_t2i",
        ("t2i", 10): "summary/test/coco_5k_r10_t2i",
    },
    "coco_cxc": {   # ECCV-corrected COCO (CxC re-annotation)
        ("i2t", 1):  "summary/test/cxc_r1_i2t",
        ("i2t", 5):  "summary/test/cxc_r5_i2t",
        ("i2t", 10): "summary/test/cxc_r10_i2t",
        ("t2i", 1):  "summary/test/cxc_r1_t2i",
        ("t2i", 5):  "summary/test/cxc_r5_t2i",
        ("t2i", 10): "summary/test/cxc_r10_t2i",
    },
}

# dataset_key → which config/dataset values qualify
DATASET_NORM: dict[str, str] = {
    "flickr30k": "flickr30k",
    "coco":      "coco",
    "coco_cxc":  "coco",
}

DATASET_LABELS = {
    "flickr30k": "Flickr30K",
    "coco":      "COCO 5K",
    "coco_cxc":  "COCO CxC (ECCV-corrected)",
}

SC_OVERALL_COL = "summary/sugarcrepe/macro_avg"
SC_OVERALL_ALT = "summary/sugarcrepe/overall"
SC_CAT_COLS    = {cat: f"summary/sugarcrepe/{cat}" for cat in SC_CATEGORIES}

N_BOOTSTRAP = 10_000
RNG_SEED    = 42


# ─── Value helpers ────────────────────────────────────────────────────────────

def to_pct(v: object) -> float:
    """Return percentage value; converts 0–1 fractions."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    v = float(v)
    if np.isnan(v):
        return np.nan
    return v * 100.0 if abs(v) <= 1.5 else v


def _safe_mean(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return float(v.mean()) if len(v) > 0 else np.nan


def _safe_std(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return float(v.std(ddof=1)) if len(v) >= 2 else np.nan


# ─── Statistical helpers ──────────────────────────────────────────────────────

def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, y = _finite_pairs(np.asarray(x, float), np.asarray(y, float))
    if len(x) < 2:
        return np.nan, np.nan
    r, p = scipy_stats.pearsonr(x, y)
    return float(r), float(p)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, y = _finite_pairs(np.asarray(x, float), np.asarray(y, float))
    if len(x) < 2:
        return np.nan, np.nan
    r, p = scipy_stats.spearmanr(x, y)
    return float(r), float(p)


def kendalltau(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, y = _finite_pairs(np.asarray(x, float), np.asarray(y, float))
    if len(x) < 2:
        return np.nan, np.nan
    t, p = scipy_stats.kendalltau(x, y)
    return float(t), float(p)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """95 % bootstrap CI for stat_fn(x, y) → (r, p)."""
    rng = np.random.default_rng(seed)
    x, y = _finite_pairs(np.asarray(x, float), np.asarray(y, float))
    if len(x) < 2:
        return np.nan, np.nan
    boots: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(x), size=len(x))
        r, _ = stat_fn(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return np.nan, np.nan
    a = np.array(boots)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["dataset_norm"] = df["config/dataset"].replace({"flickr": "flickr30k"})
    df = df[df["state"] == "finished"].copy()

    # Unified SC overall (prefer macro_avg over overall)
    overall = pd.to_numeric(df.get(SC_OVERALL_COL, pd.Series(dtype=float)), errors="coerce")
    alt     = pd.to_numeric(df.get(SC_OVERALL_ALT,  pd.Series(dtype=float)), errors="coerce")
    df["_sc_overall_raw"] = overall.combine_first(alt)

    for cat, col in SC_CAT_COLS.items():
        df[f"_sc_{cat}_raw"] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

    return df


# ─── Per-run aggregation ──────────────────────────────────────────────────────

# run_table keys: (run_id, dataset_key) → dict of metric → (mean, std, n)
RunTable = dict[tuple[str, str], dict]


def _retr_primary_col(dataset_key: str) -> str | None:
    return RETRIEVAL_SPECS.get(dataset_key, {}).get(("t2i", 1))


def build_run_table(df: pd.DataFrame) -> RunTable:
    known_runs = set(REFERENCE_MAP) | REFERENCE_RUNS
    table: RunTable = {}

    for run_id, grp in df.groupby("config/run_id", dropna=True):
        run_id = str(run_id)
        if run_id not in known_runs:
            continue

        for dataset_key, dataset_norm in DATASET_NORM.items():
            sub = grp[grp["dataset_norm"] == dataset_norm].copy()
            if sub.empty:
                continue

            retr_spec  = RETRIEVAL_SPECS[dataset_key]
            primary    = _retr_primary_col(dataset_key)

            # Require primary retrieval metric (R@1 T2I) AND SC overall
            has_retr = (
                sub[primary].notna()
                if primary and primary in sub.columns
                else pd.Series(False, index=sub.index)
            )
            has_sc = sub["_sc_overall_raw"].notna()
            both   = sub[has_retr & has_sc]

            if both.empty:
                continue

            entry: dict = {"n_seeds": len(both)}

            # SC metrics (convert to %)
            sc_vals = both["_sc_overall_raw"].apply(to_pct)
            entry["sc_overall"] = (_safe_mean(sc_vals), _safe_std(sc_vals), len(sc_vals.dropna()))
            for cat in SC_CATEGORIES:
                cat_vals = both[f"_sc_{cat}_raw"].apply(to_pct)
                entry[f"sc_{cat}"] = (_safe_mean(cat_vals), _safe_std(cat_vals), len(cat_vals.dropna()))

            # Retrieval metrics (convert to %)
            for (direction, k), col in retr_spec.items():
                vals = (
                    pd.to_numeric(both[col], errors="coerce").apply(to_pct)
                    if col in both.columns
                    else pd.Series(dtype=float)
                )
                entry[(direction, k)] = (
                    _safe_mean(vals),
                    _safe_std(vals),
                    int(vals.notna().sum()),
                )

            table[(run_id, dataset_key)] = entry

    return table


# ─── Delta computation ────────────────────────────────────────────────────────

def compute_deltas(table: RunTable) -> pd.DataFrame:
    rows = []

    for (run_id, dataset_key), entry in table.items():
        if run_id in REFERENCE_RUNS:
            continue
        ref_id = REFERENCE_MAP.get(run_id)
        if ref_id is None:
            continue

        ref = table.get((ref_id, dataset_key))

        def mean_of(key: object, src: dict | None) -> float:
            if src is None:
                return np.nan
            t = src.get(key, (np.nan,))
            return t[0] if isinstance(t, tuple) else np.nan

        def delta(key: object) -> float:
            v = mean_of(key, entry)
            r = mean_of(key, ref)
            if np.isnan(v) or np.isnan(r):
                return np.nan
            return v - r

        def avg_delta(key_a: object, key_b: object) -> float:
            da, db = delta(key_a), delta(key_b)
            if np.isnan(da) and np.isnan(db):
                return np.nan
            return float(np.nanmean([da, db]))

        n = entry["n_seeds"]
        row: dict = {
            "intervention_id":   run_id,
            "reference_id":      ref_id,
            "dataset":           dataset_key,
            "dataset_label":     DATASET_LABELS[dataset_key],
            "n_seeds":           n,
            "single_seed_flag":  n == 1,
            # absolute values
            "R1_I2T":  mean_of(("i2t", 1), entry),
            "R5_I2T":  mean_of(("i2t", 5), entry),
            "R10_I2T": mean_of(("i2t", 10), entry),
            "R1_T2I":  mean_of(("t2i", 1), entry),
            "R5_T2I":  mean_of(("t2i", 5), entry),
            "R10_T2I": mean_of(("t2i", 10), entry),
            "SC_overall": mean_of("sc_overall", entry),
        }
        for cat in SC_CATEGORIES:
            row[f"SC_{cat}"] = mean_of(f"sc_{cat}", entry)

        # deltas
        row["delta_R@1_I2T"]   = delta(("i2t", 1))
        row["delta_R@1_T2I"]   = delta(("t2i", 1))
        row["delta_R@5_avg"]   = avg_delta(("i2t", 5),  ("t2i", 5))
        row["delta_R@10_avg"]  = avg_delta(("i2t", 10), ("t2i", 10))
        row["delta_sugarcrepe_overall"] = delta("sc_overall")
        for cat in SC_CATEGORIES:
            row[f"delta_sugarcrepe_{cat}"] = delta(f"sc_{cat}")

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "intervention_id"]).reset_index(drop=True)
    return df


# ─── Correlation analysis ─────────────────────────────────────────────────────

CorrelationResult = dict[str, dict]


def run_correlations(delta_df: pd.DataFrame) -> dict[str, CorrelationResult]:
    """Pearson, Spearman, Kendall between Δ retrieval and Δ SC overall per dataset."""
    results: dict[str, CorrelationResult] = {}

    for dataset_key in ["flickr30k", "coco", "coco_cxc"]:
        sub = delta_df[delta_df["dataset"] == dataset_key].copy()
        sub = sub.dropna(subset=["delta_R@1_T2I", "delta_sugarcrepe_overall"])
        n = len(sub)
        if n < 2:
            print(f"  {DATASET_LABELS[dataset_key]}: only {n} complete pairs — skipping correlations.")
            continue

        # average R@1 across directions
        sub["delta_R@1_avg"] = sub[["delta_R@1_T2I", "delta_R@1_I2T"]].mean(axis=1)

        y = sub["delta_sugarcrepe_overall"].values
        cr: CorrelationResult = {}

        for x_name in ["delta_R@1_T2I", "delta_R@1_I2T", "delta_R@1_avg"]:
            x = sub[x_name].values
            pr, pp = pearson(x, y)
            sr, sp = spearman(x, y)
            kt, kp = kendalltau(x, y)
            pr_lo, pr_hi = bootstrap_ci(x, y, pearson)
            sr_lo, sr_hi = bootstrap_ci(x, y, spearman)
            # count valid pairs
            n_pairs = int((np.isfinite(np.asarray(x, float)) & np.isfinite(np.asarray(y, float))).sum())
            cr[x_name] = dict(
                n=n_pairs,
                pearson_r=pr, pearson_p=pp,
                pearson_ci_lo=pr_lo, pearson_ci_hi=pr_hi,
                spearman_r=sr, spearman_p=sp,
                spearman_ci_lo=sr_lo, spearman_ci_hi=sr_hi,
                kendall_tau=kt, kendall_p=kp,
            )

        results[dataset_key] = cr

    return results


def run_subcategory_spearman(
    delta_df: pd.DataFrame,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Spearman(Δ R@1 T2I, Δ SC_{cat}) per dataset."""
    out: dict[str, dict[str, tuple[float, float]]] = {}

    for dataset_key in ["flickr30k", "coco"]:
        sub = delta_df[delta_df["dataset"] == dataset_key].copy()
        x   = sub["delta_R@1_T2I"].values.astype(float)
        if np.isfinite(x).sum() < 2:
            continue
        cat_result: dict[str, tuple[float, float]] = {}
        for cat in SC_CATEGORIES:
            y = sub[f"delta_sugarcrepe_{cat}"].values.astype(float)
            r, p = spearman(x, y)
            cat_result[cat] = (r, p)
        out[dataset_key] = cat_result

    return out


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman"],
        "axes.grid":          False,
        "axes.linewidth":     0.7,
        "axes.labelsize":     9,
        "axes.titlesize":     10,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
    })


def plot_subcategory_heatmap(
    subcat_results: dict[str, dict[str, tuple[float, float]]],
    out_path: Path,
) -> None:
    datasets = [d for d in ["flickr30k", "coco"] if d in subcat_results]
    if not datasets:
        print("  Heatmap: no subcategory data available.", file=sys.stderr)
        return

    n_ds  = len(datasets)
    fig, axes = plt.subplots(
        n_ds, 1, figsize=(8, 1.8 * n_ds + 0.8), squeeze=False
    )

    cat_labels = [SC_CATEGORY_LABELS[c] for c in SC_CATEGORIES]

    for row_idx, dataset_key in enumerate(datasets):
        ax     = axes[row_idx][0]
        cr     = subcat_results[dataset_key]
        rhos   = np.array([cr.get(c, (np.nan, np.nan))[0] for c in SC_CATEGORIES])
        pvals  = np.array([cr.get(c, (np.nan, np.nan))[1] for c in SC_CATEGORIES])

        im = ax.imshow(
            rhos.reshape(1, -1), vmin=-1, vmax=1,
            cmap="RdYlGn", aspect="auto"
        )
        ax.set_xticks(range(len(SC_CATEGORIES)))
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels([DATASET_LABELS[dataset_key]], fontsize=8)

        for j, (rho, p) in enumerate(zip(rhos, pvals)):
            text  = f"{rho:.2f}" if np.isfinite(rho) else "—"
            star  = "*" if np.isfinite(p) and p < 0.05 else ""
            color = "white" if np.isfinite(rho) and abs(rho) > 0.65 else "black"
            ax.text(j, 0, text + star, ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if star else "normal")

    cbar = fig.colorbar(im, ax=axes[:, 0], label="Spearman ρ",
                        fraction=0.025, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Spearman ρ: Δ R@1 T2I vs Δ SugarCrepe subcategory\n"
        "(*p<0.05; computed across interventions per dataset)",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap → {out_path}")


def plot_quadrant_scatter(delta_df: pd.DataFrame, out_path: Path) -> None:
    primary = [d for d in ["flickr30k", "coco"]
               if d in delta_df["dataset"].values]
    if not primary:
        print("  Scatter: no data for primary datasets.", file=sys.stderr)
        return

    n = len(primary)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), squeeze=False)

    for ax, dataset_key in zip(axes[0], primary):
        sub = delta_df[delta_df["dataset"] == dataset_key].dropna(
            subset=["delta_R@1_T2I", "delta_sugarcrepe_overall"]
        )
        label = DATASET_LABELS[dataset_key]

        if sub.empty:
            ax.set_title(f"{label}\n(no data)")
            continue

        x    = sub["delta_R@1_T2I"].values.astype(float)
        y    = sub["delta_sugarcrepe_overall"].values.astype(float)
        ids  = sub["intervention_id"].values
        single = sub["single_seed_flag"].values

        xpad = max(np.nanmax(np.abs(x)) * 1.35, 0.6)
        ypad = max(np.nanmax(np.abs(y)) * 1.35, 0.6)

        # Quadrant shading
        ax.fill_between([-xpad, 0], 0, ypad,  color="#d4efdf", alpha=0.40, zorder=0)
        ax.fill_between([0, xpad],  0, ypad,  color="#a9dfbf", alpha=0.50, zorder=0)
        ax.fill_between([0, xpad],  -ypad, 0, color="#fce5cd", alpha=0.40, zorder=0)
        ax.fill_between([-xpad, 0], -ypad, 0, color="#fadbd8", alpha=0.40, zorder=0)

        # Axes
        ax.axhline(0, color="0.45", lw=0.9, ls="--", zorder=1)
        ax.axvline(0, color="0.45", lw=0.9, ls="--", zorder=1)

        # Quadrant labels (small, italic)
        def ql(tx, ty, text):
            ax.text(tx, ty, text, ha="center", va="center",
                    fontsize=6.5, color="0.40", style="italic", zorder=2)
        ql( xpad * 0.65,  ypad * 0.88, "helps both")
        ql(-xpad * 0.65,  ypad * 0.88, "SC only")
        ql( xpad * 0.65, -ypad * 0.88, "retr only")
        ql(-xpad * 0.65, -ypad * 0.88, "hurts both")

        # Points
        cmap   = plt.cm.tab20(np.linspace(0, 1, max(len(sub), 1)))
        for i, (xi, yi, lab, sg) in enumerate(zip(x, y, ids, single)):
            ax.scatter(xi, yi, color=cmap[i], s=60, zorder=4,
                       edgecolors="white", linewidth=0.6)
            suffix = "*" if sg else ""
            ax.annotate(
                lab + suffix, (xi, yi),
                textcoords="offset points", xytext=(5, 3),
                fontsize=6, zorder=5,
            )

        ax.set_xlim(-xpad, xpad)
        ax.set_ylim(-ypad, ypad)
        ax.set_xlabel("Δ R@1 T2I (pp)", fontsize=9)
        ax.set_ylabel("Δ SugarCrepe overall (pp)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Retrieval vs Compositional Discrimination (quadrant analysis)\n"
        "x = Δ R@1 T2I vs reference, y = Δ SugarCrepe overall; * = single seed",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter → {out_path}")


# ─── Correlation text report ──────────────────────────────────────────────────

def write_correlation_report(
    corr: dict[str, CorrelationResult],
    subcat: dict[str, dict[str, tuple[float, float]]],
    out_path: Path,
) -> None:
    def f(v: float) -> str:
        return f"{v:.4f}" if np.isfinite(v) else "—"

    lines: list[str] = ["# Correlation Analysis: Δ Retrieval vs Δ SugarCrepe\n"]

    for dataset_key in ["flickr30k", "coco", "coco_cxc"]:
        label = DATASET_LABELS[dataset_key]
        cr = corr.get(dataset_key)
        if cr is None:
            lines.append(f"## {label}: insufficient data (< 2 complete pairs)\n")
            continue

        lines.append(f"## {label}\n")

        for x_name, vals in cr.items():
            n = vals.get("n", "?")
            lines.append(f"### {x_name} vs Δ SC overall  (n={n} interventions)\n")
            lines.append(
                f"  Pearson:  r = {f(vals['pearson_r'])}, "
                f"p = {f(vals['pearson_p'])}, "
                f"95% CI = [{f(vals['pearson_ci_lo'])}, {f(vals['pearson_ci_hi'])}] "
                f"(10k bootstrap over interventions)\n"
            )
            lines.append(
                f"  Spearman: ρ = {f(vals['spearman_r'])}, "
                f"p = {f(vals['spearman_p'])}, "
                f"95% CI = [{f(vals['spearman_ci_lo'])}, {f(vals['spearman_ci_hi'])}]\n"
            )
            lines.append(
                f"  Kendall:  τ = {f(vals['kendall_tau'])}, "
                f"p = {f(vals['kendall_p'])}\n"
            )

        sr = subcat.get(dataset_key)
        if sr:
            lines.append(f"### Subcategory Spearman (x = Δ R@1 T2I)\n")
            for cat, (rho, p) in sorted(
                sr.items(), key=lambda kv: abs(kv[1][0]) if np.isfinite(kv[1][0]) else 0,
                reverse=True,
            ):
                star = " *" if np.isfinite(p) and p < 0.05 else ""
                lines.append(f"  {cat:<18}: ρ = {f(rho)}, p = {f(p)}{star}\n")
        lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Saved correlation report → {out_path}")


# ─── Markdown summary ─────────────────────────────────────────────────────────

def write_summary(
    delta_df: pd.DataFrame,
    corr: dict[str, CorrelationResult],
    subcat: dict[str, dict[str, tuple[float, float]]],
    out_path: Path,
) -> None:
    def f2(v: float) -> str:
        return f"{v:.2f}" if np.isfinite(v) else "—"

    def quadrant_ids(df_sub: pd.DataFrame) -> dict[str, list[str]]:
        return {
            "helps_both": df_sub[
                (df_sub["delta_R@1_T2I"] > 0) & (df_sub["delta_sugarcrepe_overall"] > 0)
            ]["intervention_id"].tolist(),
            "retr_only": df_sub[
                (df_sub["delta_R@1_T2I"] > 0) & (df_sub["delta_sugarcrepe_overall"] <= 0)
            ]["intervention_id"].tolist(),
            "sc_only": df_sub[
                (df_sub["delta_R@1_T2I"] <= 0) & (df_sub["delta_sugarcrepe_overall"] > 0)
            ]["intervention_id"].tolist(),
            "hurts_both": df_sub[
                (df_sub["delta_R@1_T2I"] <= 0) & (df_sub["delta_sugarcrepe_overall"] <= 0)
            ]["intervention_id"].tolist(),
        }

    single_seeds = int(delta_df["single_seed_flag"].sum())
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append(f"# SugarCrepe–Retrieval Correspondence: Summary\n\n")
    lines.append(f"_Generated: {ts}_\n\n")

    # ── Overall correlation narrative ─────────────────────────────────────────
    lines.append("## 1. Retrieval–Composition Correlation\n\n")

    for dataset_key in ["flickr30k", "coco"]:
        label = DATASET_LABELS[dataset_key]
        cr    = corr.get(dataset_key, {})
        r1t2i = cr.get("delta_R@1_T2I", {})
        sr    = r1t2i.get("spearman_r", np.nan)
        sr_lo = r1t2i.get("spearman_ci_lo", np.nan)
        sr_hi = r1t2i.get("spearman_ci_hi", np.nan)
        pr    = r1t2i.get("pearson_r", np.nan)
        kt    = r1t2i.get("kendall_tau", np.nan)
        n     = r1t2i.get("n", "?")

        if not np.isfinite(sr):
            lines.append(f"**{label}**: insufficient data for correlation.\n\n")
            continue

        direction = (
            "positive" if sr > 0.2 else
            "negative" if sr < -0.2 else
            "near-zero (decoupled)"
        )
        lines.append(
            f"**{label}** (n={n} interventions with complete retrieval+SugarCrepe): "
            f"Δ R@1 T2I and Δ SugarCrepe overall are **{direction}** "
            f"(Spearman ρ = {f2(sr)}, 95 % CI [{f2(sr_lo)}, {f2(sr_hi)}]; "
            f"Pearson r = {f2(pr)}; Kendall τ = {f2(kt)}). "
        )
        if abs(sr) < 0.3:
            lines.append(
                "The wide CI and near-zero point estimate indicate the two "
                "objectives are largely decoupled at this ladder scale.\n\n"
            )
        elif sr > 0.6:
            lines.append(
                "The strong positive association suggests that interventions "
                "that help retrieval tend also to improve compositional "
                "discrimination.\n\n"
            )
        elif sr < -0.4:
            lines.append(
                "The negative association suggests a tension: interventions "
                "that boost retrieval tend to hurt compositional discrimination "
                "and vice versa.\n\n"
            )
        else:
            lines.append("\n\n")

    # ── Quadrant breakdown ────────────────────────────────────────────────────
    lines.append("## 2. Quadrant Breakdown\n\n")

    for dataset_key in ["flickr30k", "coco"]:
        label = DATASET_LABELS[dataset_key]
        sub   = delta_df[delta_df["dataset"] == dataset_key].dropna(
            subset=["delta_R@1_T2I", "delta_sugarcrepe_overall"]
        )
        if sub.empty:
            continue
        q = quadrant_ids(sub)
        lines.append(f"**{label}**\n\n")
        lines.append(f"- Helps both (ΔR@1>0 & ΔSC>0): "
                     f"{', '.join(q['helps_both']) if q['helps_both'] else '—'}\n")
        lines.append(f"- Retrieval-only (ΔR@1>0, ΔSC≤0): "
                     f"{', '.join(q['retr_only']) if q['retr_only'] else '—'}\n")
        lines.append(f"- SugarCrepe-only (ΔR@1≤0, ΔSC>0): "
                     f"{', '.join(q['sc_only']) if q['sc_only'] else '—'}\n")
        lines.append(f"- Hurts both (ΔR@1≤0, ΔSC≤0): "
                     f"{', '.join(q['hurts_both']) if q['hurts_both'] else '—'}\n\n")

    # ── Subcategory notes ─────────────────────────────────────────────────────
    lines.append("## 3. Subcategory Tracking\n\n")

    for dataset_key in ["flickr30k", "coco"]:
        label = DATASET_LABELS[dataset_key]
        sr    = subcat.get(dataset_key)
        if not sr:
            continue
        ordered = sorted(sr.items(),
                         key=lambda kv: abs(kv[1][0]) if np.isfinite(kv[1][0]) else 0,
                         reverse=True)
        top    = [c for c, (r, p) in ordered if np.isfinite(r) and abs(r) > 0.4]
        bottom = [c for c, (r, p) in ordered if np.isfinite(r) and abs(r) < 0.2]

        lines.append(f"**{label}**: subcategories most closely tracking retrieval: "
                     f"{', '.join(top) if top else '—'}. "
                     f"Most decoupled: {', '.join(bottom) if bottom else '—'}.\n\n")

    # ── Dataset comparison ────────────────────────────────────────────────────
    lines.append("## 4. Flickr30K vs COCO\n\n")
    sr_flk  = corr.get("flickr30k",  {}).get("delta_R@1_T2I", {}).get("spearman_r",  np.nan)
    sr_coco = corr.get("coco",        {}).get("delta_R@1_T2I", {}).get("spearman_r",  np.nan)

    if np.isfinite(sr_flk) and np.isfinite(sr_coco):
        diff = abs(sr_flk - sr_coco)
        if diff < 0.2:
            lines.append(
                f"The correlation pattern is **consistent** across datasets "
                f"(Flickr ρ={f2(sr_flk)}, COCO ρ={f2(sr_coco)}), "
                f"suggesting a dataset-agnostic relationship between the two objectives.\n\n"
            )
        else:
            lines.append(
                f"The correlation differs between datasets "
                f"(Flickr ρ={f2(sr_flk)}, COCO ρ={f2(sr_coco)}). "
                f"The picture may depend on dataset-specific factors (e.g. "
                f"COCO's denser label space favoring composition, Flickr's "
                f"retrieval difficulty profile).\n\n"
            )
    else:
        lines.append("Insufficient data to compare datasets.\n\n")

    # ── Data caveats ──────────────────────────────────────────────────────────
    lines.append("## 5. Caveats\n\n")
    lines.append(
        f"- **Single-seed runs** (marked * in figures): {single_seeds} intervention–dataset "
        f"pairs have n=1 seed; these are included in point estimates but **excluded** "
        f"from bootstrap CI computation (CIs are computed over interventions, not seeds).\n"
        f"- Bootstrap CIs use 10 000 resamples over the n interventions with complete data; "
        f"they reflect sampling uncertainty in the correlation statistic given the observed "
        f"set of interventions, not seed-level variance.\n"
        f"- **B0plus_fixed** is used as the B0+ reference "
        f"(it is the `parent:` entry in registry.yaml for all B5x/BLIP_TEXT runs); "
        f"B0plus itself has no SugarCrepe evaluations.\n"
        f"- COCO CxC (ECCV-corrected) rows are tabulated separately but not included in "
        f"the primary correlation analysis (use delta\\_table.csv for those values).\n"
        f"- Runs missing either retrieval or SugarCrepe metrics are excluded entirely.\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Saved summary → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path("analysis/sugarcrepe_correspondence") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    setup_style()

    # 1. Load
    print("Loading W&B CSV …")
    df = load_raw()
    print(f"  Finished runs: {len(df)}\n")

    # 2. Aggregate
    print("Aggregating per-run metrics …")
    table = build_run_table(df)
    print(f"  Entries (run_id × dataset) with both retrieval + SC: {len(table)}")
    for (rid, dset), entry in sorted(table.items()):
        n = entry["n_seeds"]
        sc = entry.get("sc_overall", (np.nan,))[0]
        r1 = entry.get(("t2i", 1), (np.nan,))[0]
        flag = " *" if n == 1 else ""
        print(f"    {rid:<30} {DATASET_LABELS.get(dset, dset):<25} "
              f"n={n}{flag}  SC={sc:.1f}%  R1T2I={r1:.2f}%")
    print()

    # 3. Deltas
    print("Computing deltas vs reference …")
    delta_df = compute_deltas(table)
    print(f"  Delta rows: {len(delta_df)}\n")

    if delta_df.empty:
        print("ERROR: no delta rows — check data / run_ids.", file=sys.stderr)
        sys.exit(1)

    print(delta_df[
        ["intervention_id", "dataset", "n_seeds",
         "delta_R@1_T2I", "delta_sugarcrepe_overall"]
    ].to_string(index=False))
    print()

    # ── Output 1: long-form table ──────────────────────────────────────────────
    tbl_cols = (
        ["intervention_id", "reference_id", "dataset", "dataset_label",
         "n_seeds", "single_seed_flag",
         "delta_R@1_I2T", "delta_R@1_T2I", "delta_R@5_avg", "delta_R@10_avg",
         "delta_sugarcrepe_overall"]
        + [f"delta_sugarcrepe_{cat}" for cat in SC_CATEGORIES]
        + ["R1_I2T", "R1_T2I", "R5_I2T", "R5_T2I", "R10_I2T", "R10_T2I",
           "SC_overall"]
        + [f"SC_{cat}" for cat in SC_CATEGORIES]
    )
    tbl_out  = delta_df[[c for c in tbl_cols if c in delta_df.columns]]
    tbl_path = out_dir / "delta_table.csv"
    tbl_out.to_csv(tbl_path, index=False, float_format="%.4f")
    print(f"Saved delta table → {tbl_path}")

    # ── Outputs 2–5 require scipy ──────────────────────────────────────────────
    corr_results: dict[str, CorrelationResult] = {}
    subcat_results: dict[str, dict[str, tuple[float, float]]] = {}

    if HAS_SCIPY:
        print("\nRunning correlation analysis …")
        corr_results   = run_correlations(delta_df)
        subcat_results = run_subcategory_spearman(delta_df)

        write_correlation_report(
            corr_results, subcat_results,
            out_dir / "correlation_analysis.txt",
        )
    else:
        print("\nskipping correlation analysis (scipy not available)")

    print("\nGenerating figures …")
    plot_subcategory_heatmap(subcat_results, out_dir / "subcategory_heatmap.png")
    plot_quadrant_scatter(delta_df, out_dir / "quadrant_scatter.png")

    print("\nWriting summary …")
    write_summary(delta_df, corr_results, subcat_results, out_dir / "summary.md")

    print(f"\nDone. All outputs → {out_dir}/")


if __name__ == "__main__":
    main()
