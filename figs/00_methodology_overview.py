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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from helpers import SAVE_FIG_DIR


COLORS = {
    "default": "#E8E8E8",
    "loss": "#9ECAE1",
    "data": "#A1D99B",
    "arch": "#FDBF6F",
    "segment": "#BCBDDC",
    "frozen": "#D9D9D9",
}
EDGE = "#333333"


def box(ax, xy, wh, text, fc, dashed=False, fontsize=7.2, lw=1.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor=fc,
        edgecolor=EDGE,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True, zorder=3)
    return patch


def arrow(ax, start, end, dashed=False, color=EDGE, rad=0.0, lw=1.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)
    return arr


def elbow_arrow(ax, points, dashed=False, color=EDGE, lw=1.0):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            solid_capstyle="round",
            zorder=1,
        )
    return arrow(ax, points[-2], points[-1], dashed=dashed, color=color, lw=lw)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.45, 6.90, "Default CLIP retrieval path", fontsize=11, weight="bold", va="top")
    ax.text(
        0.45,
        6.60,
        "Dashed boxes/arrows are optional interventions or diagnostics.",
        fontsize=8,
        color="0.28",
        va="top",
    )

    # Inputs and data augmentations
    box(ax, (0.45, 4.55), (1.35, 0.58), "Image x", COLORS["default"], fontsize=8.8)
    box(ax, (0.45, 2.50), (1.35, 0.58), "Caption c", COLORS["default"], fontsize=8.8)
    box(ax, (2.25, 5.08), (1.55, 0.70), "Augmented\nview x'\nCrop / Color", COLORS["data"], dashed=True, fontsize=7.2)
    box(ax, (2.25, 1.58), (1.55, 0.70), "LLM\nparaphrase c'", COLORS["data"], dashed=True, fontsize=7.2)
    arrow(ax, (1.80, 4.88), (2.25, 5.34), dashed=True, color="#238B45")
    arrow(ax, (1.80, 2.78), (2.25, 1.94), dashed=True, color="#238B45")

    # Encoders
    box(
        ax,
        (4.10, 4.00),
        (2.20, 0.92),
        "CLIP ViT-L/14@336\nImage Encoder\nunfreeze last 4 blocks\n+ visual projection",
        COLORS["default"],
        fontsize=7.4,
    )
    box(
        ax,
        (4.10, 2.55),
        (2.20, 0.78),
        "CLIP Text Encoder\nfrozen body\ntext projection trained",
        COLORS["frozen"],
        fontsize=7.4,
    )
    box(
        ax,
        (4.10, 1.10),
        (2.20, 0.78),
        "BLIP Text Encoder\nfrozen diagnostic\n+ trainable text proj",
        COLORS["arch"],
        dashed=True,
        fontsize=7.2,
    )
    arrow(ax, (1.80, 4.84), (4.10, 4.46))
    arrow(ax, (3.80, 5.43), (4.10, 4.78), dashed=True, color="#238B45")
    arrow(ax, (1.80, 2.79), (4.10, 2.94))
    elbow_arrow(ax, [(3.80, 1.93), (3.98, 1.93), (4.10, 1.50)], dashed=True, color="#D95F02")

    # Segment-aware visual branch
    box(
        ax,
        (4.18, 5.18),
        (1.80, 0.70),
        "SAM inputs\nspatial / semantic\ncontinuous",
        COLORS["segment"],
        dashed=True,
        fontsize=7.0,
    )
    box(
        ax,
        (6.38, 5.04),
        (1.62, 0.78),
        "Patch-token\ninjection\nbefore ViT",
        COLORS["segment"],
        dashed=True,
        fontsize=7.0,
    )
    box(
        ax,
        (6.52, 3.10),
        (1.58, 0.78),
        "Multi-stream\nSAM fusion\ngate / cross-attn",
        COLORS["segment"],
        dashed=True,
        fontsize=7.0,
    )
    arrow(ax, (5.98, 5.52), (6.38, 5.44), dashed=True, color="#756BB1")
    elbow_arrow(ax, [(7.19, 5.04), (7.19, 4.88), (6.30, 4.68)], dashed=True, color="#756BB1")
    arrow(ax, (6.30, 4.28), (6.52, 3.50), dashed=True, color="#756BB1", rad=-0.12)

    # Projections
    box(ax, (8.38, 4.10), (1.42, 0.78), "Image\nProjection\n768-d default\n1024-d ablation", COLORS["arch"], fontsize=7.0)
    box(ax, (8.38, 2.55), (1.42, 0.78), "Text\nProjection\n768-d default\n1024-d ablation", COLORS["arch"], fontsize=7.0)
    arrow(ax, (6.30, 4.46), (8.38, 4.49))
    elbow_arrow(ax, [(8.10, 3.49), (8.26, 3.49), (8.38, 4.30)], dashed=True, color="#756BB1")
    arrow(ax, (6.30, 2.94), (8.38, 2.94))
    arrow(ax, (6.30, 1.49), (8.38, 2.73), dashed=True, color="#D95F02", rad=0.18)

    # Losses
    box(ax, (10.20, 4.06), (1.55, 0.90), "Image-Text\nContrastive\nInfoNCE", COLORS["loss"], fontsize=7.3)
    box(ax, (10.20, 5.26), (1.55, 0.58), "SigLIP\nalternative", COLORS["loss"], dashed=True, fontsize=7.2)
    box(ax, (10.20, 3.30), (1.55, 0.52), "Image-Image\nintra-modal", COLORS["loss"], dashed=True, fontsize=6.8)
    box(ax, (10.20, 2.62), (1.55, 0.52), "Text-Text\nintra-modal", COLORS["loss"], dashed=True, fontsize=6.8)
    box(ax, (10.20, 1.94), (1.55, 0.52), "Hard negative\ncaptions", COLORS["loss"], dashed=True, fontsize=6.8)
    box(ax, (10.20, 1.26), (1.55, 0.52), "Object\nclassification head", COLORS["loss"], dashed=True, fontsize=6.4)
    arrow(ax, (9.80, 4.49), (10.20, 4.62))
    elbow_arrow(ax, [(9.80, 2.94), (10.02, 2.94), (10.02, 4.35), (10.20, 4.35)])
    arrow(ax, (9.80, 4.72), (10.20, 5.50), dashed=True, color="#3182BD", rad=0.18)
    elbow_arrow(ax, [(9.80, 4.35), (10.02, 4.35), (10.02, 3.56), (10.20, 3.56)], dashed=True, color="#3182BD")
    arrow(ax, (9.80, 2.94), (10.20, 2.88), dashed=True, color="#3182BD")
    arrow(ax, (9.80, 2.73), (10.20, 2.20), dashed=True, color="#3182BD", rad=0.12)
    elbow_arrow(ax, [(6.30, 4.10), (8.10, 4.10), (8.10, 1.52), (10.20, 1.52)], dashed=True, color="#3182BD")

    handles = [
        Patch(facecolor=COLORS["loss"], edgecolor=EDGE, label="Loss interventions"),
        Patch(facecolor=COLORS["data"], edgecolor=EDGE, label="Data interventions"),
        Patch(facecolor=COLORS["arch"], edgecolor=EDGE, label="Architectural ablations"),
        Patch(facecolor=COLORS["segment"], edgecolor=EDGE, label="Segment-aware"),
        Patch(facecolor=COLORS["default"], edgecolor=EDGE, label="Default / frozen"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, -0.11), frameon=False, fontsize=6.5)

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / "00_methodology_overview.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SAVE_FIG_DIR / "00_methodology_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
