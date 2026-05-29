from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "cache" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from helpers import SAVE_FIG_DIR, setup_thesis_style


EDGE = "#2F2F2F"
TRAINABLE = "#FDBF6F"
FROZEN = "#D9D9D9"
SAM = "#BCBDDC"
TOKEN = "#E8E8E8"
OUT = "#BFD8EA"
ARROW = "#4B4B4B"


def box(ax, xy, wh, text, fc, ec=EDGE, dashed=False, fontsize=6.8, lw=1.0, zorder=2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, color=ARROW, dashed=False, lw=1.0, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def lock_icon(ax, x, y, scale=1.0):
    ax.add_patch(
        Rectangle(
            (x - 0.040 * scale, y - 0.040 * scale),
            0.080 * scale,
            0.060 * scale,
            facecolor="#666666",
            edgecolor="#666666",
            linewidth=0.6,
            zorder=5,
        )
    )
    arc = plt.Circle((x, y + 0.020 * scale), 0.045 * scale, fill=False, color="#666666", linewidth=0.7, zorder=5)
    ax.add_patch(arc)


def panel_frame(ax, x0, title, eqno):
    ax.add_patch(
        FancyBboxPatch(
            (x0, 0.50),
            3.70,
            5.58,
            boxstyle="round,pad=0.020,rounding_size=0.045",
            facecolor="white",
            edgecolor="#BDBDBD",
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(x0 + 1.85, 5.86, title, ha="center", va="center", fontsize=9.0, weight="bold")
    ax.text(x0 + 3.48, 0.72, eqno, ha="right", va="bottom", fontsize=7.0, color="0.35")


def common_top(ax, x0):
    box(ax, (x0 + 0.22, 5.20), (0.88, 0.48), r"CLIP CLS" "\n" r"$c\in\mathbb{R}^{1024}$", TOKEN, fontsize=6.2)
    box(
        ax,
        (x0 + 2.00, 5.18),
        (1.30, 0.54),
        "frozen SAM\n" r"$8{\times}8{\times}256$",
        SAM,
        dashed=True,
        fontsize=5.9,
    )
    lock_icon(ax, x0 + 3.17, 5.62, scale=0.9)
    box(ax, (x0 + 2.04, 4.42), (1.18, 0.46), r"$W_{\mathrm{SAM}}$" "\n" r"$256\!\to\!1024$", TRAINABLE, fontsize=6.2)
    box(ax, (x0 + 2.04, 3.74), (1.18, 0.46), r"SAM tokens" "\n" r"$M\in\mathbb{R}^{64\times1024}$", TOKEN, fontsize=5.8)
    arrow(ax, (x0 + 2.65, 5.18), (x0 + 2.65, 4.88))
    arrow(ax, (x0 + 2.65, 4.42), (x0 + 2.65, 4.20))


def common_bottom(ax, x0):
    box(ax, (x0 + 1.02, 1.22), (1.42, 0.52), "CLIP visual projection\n" r"$1024\!\to\!768$", TRAINABLE, fontsize=6.2)
    box(ax, (x0 + 2.74, 1.22), (0.76, 0.52), "retrieval\nembedding", OUT, fontsize=5.8)
    ax.text(x0 + 2.66, 0.86, "retrieval loss", ha="center", va="center", fontsize=6.1, color="0.28")
    arrow(ax, (x0 + 2.44, 1.48), (x0 + 2.74, 1.48))
    arrow(ax, (x0 + 3.12, 1.22), (x0 + 2.82, 0.96), color="#08519C")


def panel_gate(ax, x0):
    common_top(ax, x0)
    common_bottom(ax, x0)
    box(ax, (x0 + 2.18, 3.05), (0.94, 0.42), r"mean-pool $M$" "\n" r"$\bar{m}$", TRAINABLE, fontsize=5.8)
    box(ax, (x0 + 0.76, 3.04), (1.06, 0.42), r"$[c;\bar{m}]$", TOKEN, fontsize=6.8)
    box(ax, (x0 + 0.76, 2.42), (1.06, 0.42), r"gate linear" "\n" r"$W_g$", TRAINABLE, fontsize=6.0)
    box(ax, (x0 + 2.10, 2.42), (0.72, 0.42), r"sigmoid" "\n" r"$g$", TRAINABLE, fontsize=6.0)
    box(ax, (x0 + 0.48, 1.92), (2.62, 0.35), r"$c'=(1-g)\odot c+g\odot\bar{m}$", TOKEN, fontsize=6.0)
    ax.text(x0 + 0.42, 0.98, r"$W_g$ zero-init $\to g_0=0.5$ (equal mix)", fontsize=5.9, color="0.30")
    arrow(ax, (x0 + 2.65, 3.74), (x0 + 2.65, 3.47))
    arrow(ax, (x0 + 2.18, 3.25), (x0 + 1.82, 3.25))
    arrow(ax, (x0 + 1.10, 5.44), (x0 + 1.22, 3.46), rad=0.08)
    arrow(ax, (x0 + 1.29, 3.04), (x0 + 1.29, 2.84))
    arrow(ax, (x0 + 1.82, 2.63), (x0 + 2.10, 2.63))
    arrow(ax, (x0 + 2.46, 2.42), (x0 + 2.14, 2.27), rad=0.05)
    arrow(ax, (x0 + 1.79, 1.92), (x0 + 1.73, 1.74))


def panel_xattn(ax, x0):
    common_top(ax, x0)
    common_bottom(ax, x0)
    box(ax, (x0 + 0.74, 3.38), (1.20, 0.50), "multi-head attention\nQ=c, K/V=M", TRAINABLE, fontsize=5.8)
    box(ax, (x0 + 0.74, 2.76), (1.20, 0.38), r"residual add" "\n" r"$c+\mathrm{MHA}(\cdot)$", TOKEN, fontsize=5.7)
    box(ax, (x0 + 2.12, 2.76), (0.82, 0.38), "LayerNorm", TRAINABLE, fontsize=6.0)
    box(ax, (x0 + 1.36, 2.10), (0.84, 0.38), r"$c'$", TOKEN, fontsize=7.0)
    ax.text(x0 + 0.34, 0.95, "no positional encoding on SAM tokens,\npost-LN residual", fontsize=5.8, color="0.30")
    arrow(ax, (x0 + 1.10, 5.44), (x0 + 1.18, 3.88), rad=0.05)
    arrow(ax, (x0 + 2.04, 3.95), (x0 + 1.94, 3.62), rad=0.10)
    arrow(ax, (x0 + 1.34, 3.38), (x0 + 1.34, 3.14))
    arrow(ax, (x0 + 1.94, 2.95), (x0 + 2.12, 2.95))
    arrow(ax, (x0 + 2.52, 2.76), (x0 + 2.20, 2.31), rad=0.08)
    arrow(ax, (x0 + 1.78, 2.10), (x0 + 1.73, 1.74))


def panel_concat(ax, x0):
    common_top(ax, x0)
    common_bottom(ax, x0)
    box(ax, (x0 + 2.18, 3.05), (0.94, 0.42), r"mean-pool $M$" "\n" r"$\bar{m}$", TRAINABLE, fontsize=5.8)
    box(ax, (x0 + 0.76, 3.04), (1.06, 0.42), r"$[c;\bar{m}]$", TOKEN, fontsize=6.8)
    box(ax, (x0 + 0.56, 2.34), (1.86, 0.48), "MLP 2048→1024→1024\nGELU between layers", TRAINABLE, fontsize=5.8)
    box(ax, (x0 + 1.10, 1.94), (0.80, 0.34), r"$c'$", TOKEN, fontsize=7.0)
    ax.text(x0 + 0.34, 0.95, r"no residual to $c$, no dropout, no LN", fontsize=5.9, color="0.30")
    arrow(ax, (x0 + 2.65, 3.74), (x0 + 2.65, 3.47))
    arrow(ax, (x0 + 2.18, 3.25), (x0 + 1.82, 3.25))
    arrow(ax, (x0 + 1.10, 5.44), (x0 + 1.22, 3.46), rad=0.08)
    arrow(ax, (x0 + 1.29, 3.04), (x0 + 1.29, 2.82))
    arrow(ax, (x0 + 1.50, 2.34), (x0 + 1.50, 2.28))
    arrow(ax, (x0 + 1.50, 1.94), (x0 + 1.73, 1.74))


def main() -> None:
    setup_thesis_style()
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.30, 6.72, "SAM fusion variants.", fontsize=10.6, weight="bold", va="top")
    ax.text(
        0.30,
        6.35,
        "Frozen 8×8 SAM features are projected and fused into the CLIP CLS path before retrieval projection.",
        fontsize=7.2,
        color="0.30",
    )

    starts = [0.25, 4.15, 8.05]
    specs = [("Sam-Gate", "(3.16)", panel_gate), ("Sam-XAttn", "(3.17)", panel_xattn), ("Sam-Concat", "(3.18)", panel_concat)]
    for x0, (title, eqno, fn) in zip(starts, specs):
        panel_frame(ax, x0, title, eqno)
        fn(ax, x0)

    ax.text(
        6.0,
        0.24,
        r"All three replace $c$ as input to the CLIP visual projection.",
        ha="center",
        va="center",
        fontsize=7.2,
        color="0.25",
    )

    fig.savefig(SAVE_FIG_DIR / "12_sam_fusion_variants.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(SAVE_FIG_DIR / "12_sam_fusion_variants.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
