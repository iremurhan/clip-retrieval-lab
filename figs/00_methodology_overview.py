from __future__ import annotations

import os
import textwrap
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "cache" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
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
GRAY_EDGE = "#6F6F6F"
TEXT_ARROW = "#2171B5"
IMAGE_ARROW = "#6BAED6"
MIXED_ARROW = "#08519C"
DATA_ARROW = "#238B45"
SEG_ARROW = "#756BB1"
ARCH_ARROW = "#D95F02"
THUMBNAIL_PATH = Path("/Volumes/T7/Research/experiments/datasets/coco/val2014/COCO_val2014_000000391895.jpg")
CAPTION_TEXT = "A man in a red shirt and a red hat is on a motorcycle on a hill side."
LABEL_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2}


def box(ax, xy, wh, text, fc, dashed=False, fontsize=7.2, lw=1.0, ec=EDGE, ha="center"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if ha == "center" else x + 0.08
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, wrap=True, zorder=3)
    return patch


def input_image_box(ax, xy, wh):
    x, y = xy
    w, h = wh
    patch = box(ax, xy, wh, "", "white", fontsize=6, lw=0.9, ec=GRAY_EDGE)
    image_area = (x + 0.08, x + w - 0.08, y + 0.30, y + h - 0.08)
    if THUMBNAIL_PATH.exists():
        img = mpimg.imread(THUMBNAIL_PATH)
        xmin, xmax, ymin, ymax = image_area
        area_w = xmax - xmin
        area_h = ymax - ymin
        img_h, img_w = img.shape[:2]
        img_aspect = img_w / img_h
        area_aspect = area_w / area_h
        if area_aspect > img_aspect:
            display_h = area_h
            display_w = display_h * img_aspect
        else:
            display_w = area_w
            display_h = display_w / img_aspect
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        ax.imshow(
            img,
            extent=(cx - display_w / 2, cx + display_w / 2, cy - display_h / 2, cy + display_h / 2),
            aspect="auto",
            interpolation="lanczos",
            zorder=2.5,
            clip_path=patch,
            clip_on=True,
        )
    else:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.08, y + 0.28),
                w - 0.16,
                h - 0.36,
                boxstyle="round,pad=0.01,rounding_size=0.018",
                facecolor="#D9D9D9",
                edgecolor="none",
                zorder=2.5,
            )
        )
    ax.text(x + w / 2, y + 0.13, "Image $x_i$", ha="center", va="center", fontsize=6.2, family="monospace", zorder=3)
    return patch


def input_caption_box(ax, xy, wh):
    x, y = xy
    w, h = wh
    box(ax, xy, wh, "", "white", fontsize=6, lw=0.9, ec=GRAY_EDGE)
    wrapped_caption = "\n".join(textwrap.wrap(CAPTION_TEXT, width=28, break_long_words=False))
    ax.text(x + 0.10, y + h - 0.17, wrapped_caption, ha="left", va="top", fontsize=5.8, zorder=3)
    ax.text(x + 0.10, y + 0.14, "Caption $c_i$", ha="left", va="center", fontsize=6.1, family="monospace", zorder=3)


def tag(ax, x, y, text, color="0.25"):
    ax.text(x, y, text, ha="left", va="center", fontsize=5.8, color=color, bbox=LABEL_BBOX, zorder=6)


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
    fig, ax = plt.subplots(figsize=(14.2, 7.2))
    ax.set_xlim(0, 16.3)
    ax.set_ylim(0, 7.7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Inputs and data augmentations
    input_image_box(ax, (0.45, 4.82), (2.05, 1.35))
    input_caption_box(ax, (0.45, 1.74), (2.05, 1.34))

    box(ax, (3.15, 6.10), (1.92, 0.54), "Augmented view $\\tilde{x}_i^{(a)}$\nCrop / Color", COLORS["data"], dashed=True, fontsize=6.7)
    box(ax, (3.15, 5.18), (1.92, 0.50), "Augmented view $\\tilde{x}_i^{(b)}$\nCrop / Color", COLORS["data"], dashed=True, fontsize=6.5)
    box(ax, (3.15, 3.00), (1.92, 0.50), "LLM paraphrase $\\tilde{c}_i^{(a)}$", COLORS["data"], dashed=True, fontsize=6.8)
    box(ax, (3.15, 2.12), (1.92, 0.50), "LLM paraphrase $\\tilde{c}_i^{(b)}$", COLORS["data"], dashed=True, fontsize=6.7)
    arrow(ax, (2.50, 5.92), (3.15, 6.37), dashed=True, color=DATA_ARROW)
    arrow(ax, (2.50, 5.42), (3.15, 5.43), dashed=True, color=DATA_ARROW)
    arrow(ax, (2.50, 2.62), (3.15, 3.26), dashed=True, color=DATA_ARROW)
    arrow(ax, (2.50, 2.26), (3.15, 2.37), dashed=True, color=DATA_ARROW)

    # Encoders
    box(
        ax,
        (5.22, 5.22),
        (2.34, 0.96),
        "CLIP ViT-L/14@336\nImage Encoder\nunfreeze last 4 blocks\n+ visual projection",
        COLORS["default"],
        fontsize=7.1,
    )
    box(
        ax,
        (5.22, 3.25),
        (2.34, 0.78),
        "CLIP Text Encoder\nfrozen body\ntext projection trained",
        COLORS["frozen"],
        fontsize=7.1,
    )
    box(
        ax,
        (5.22, 2.12),
        (2.34, 0.76),
        "BLIP Text Encoder\nfrozen diagnostic\n+ trainable text proj",
        COLORS["arch"],
        dashed=True,
        fontsize=6.9,
    )
    elbow_arrow(ax, [(2.50, 5.05), (2.86, 4.82), (5.02, 4.82), (5.22, 5.46)], color=EDGE, lw=1.15)
    arrow(ax, (5.07, 6.37), (5.22, 5.98), dashed=True, color=DATA_ARROW)
    arrow(ax, (5.07, 5.43), (5.22, 5.44), dashed=True, color=DATA_ARROW)
    elbow_arrow(ax, [(2.50, 2.90), (2.72, 3.72), (5.22, 3.72)], color=EDGE, lw=1.15)
    arrow(ax, (5.07, 3.25), (5.22, 3.52), dashed=True, color=DATA_ARROW)
    arrow(ax, (5.07, 2.37), (5.22, 3.34), dashed=True, color=DATA_ARROW)
    elbow_arrow(ax, [(2.50, 1.98), (2.72, 1.40), (4.96, 1.40), (5.22, 2.32)], dashed=True, color=ARCH_ARROW)

    # Segment-aware visual branch
    box(
        ax,
        (5.35, 6.65),
        (2.00, 0.58),
        "SAM mask → Seg-Spatial /\nSeg-Semantic / Seg-Geom",
        COLORS["segment"],
        dashed=True,
        fontsize=6.6,
    )
    box(
        ax,
        (7.95, 6.56),
        (2.00, 0.72),
        "Patch-token injection\n(additive, before pre-LN,\nCLS excluded)",
        COLORS["segment"],
        dashed=True,
        fontsize=6.25,
    )
    box(
        ax,
        (7.95, 4.45),
        (2.18, 0.80),
        "Multi-stream SAM fusion:\nSam-Gate / Sam-XAttn /\nSam-Concat",
        COLORS["segment"],
        dashed=True,
        fontsize=6.25,
    )
    ax.text(8.05, 4.33, "× frozen SAM ViT-B\n(precomputed 8×8 cache)", ha="left", va="top", fontsize=5.6, color="0.32", zorder=4)
    arrow(ax, (7.35, 6.94), (7.95, 6.93), dashed=True, color=SEG_ARROW)
    elbow_arrow(ax, [(8.95, 6.56), (8.95, 6.18), (7.56, 5.88)], dashed=True, color=SEG_ARROW)

    # Projections
    box(ax, (10.75, 5.34), (1.50, 0.66), "Image Projection\n768-d", COLORS["arch"], fontsize=7.0)
    box(ax, (10.75, 3.28), (1.50, 0.66), "Text Projection\n768-d", COLORS["arch"], fontsize=7.0)
    box(ax, (10.48, 4.36), (1.98, 0.52), "ablation: Proj-1024 inserts\nan additional 768→1024 head", COLORS["arch"], dashed=True, fontsize=5.9, lw=0.8)
    arrow(ax, (7.56, 5.70), (10.75, 5.67), color=EDGE)
    elbow_arrow(ax, [(7.56, 5.44), (7.72, 5.44), (7.72, 4.85), (7.95, 4.85)], dashed=True, color=SEG_ARROW)
    arrow(ax, (10.13, 4.85), (10.75, 5.50), dashed=True, color=SEG_ARROW, rad=0.08)
    arrow(ax, (7.56, 3.64), (10.75, 3.61), color=EDGE)
    arrow(ax, (7.56, 2.48), (10.75, 3.42), dashed=True, color=ARCH_ARROW, rad=0.15)

    # Losses
    loss_x, loss_w = 14.05, 1.48
    box(ax, (loss_x, 5.62), (loss_w, 0.64), "Image-Text\nContrastive InfoNCE", COLORS["loss"], fontsize=6.7)
    box(ax, (loss_x, 4.75), (loss_w, 0.58), "SigLIP alternative", COLORS["loss"], dashed=True, fontsize=6.5)
    tag(ax, loss_x + 0.12, 4.62, "replaces InfoNCE", color="0.28")
    box(ax, (loss_x, 3.92), (loss_w, 0.58), "Image-Image\nintra-modal", COLORS["loss"], dashed=True, fontsize=6.4)
    box(ax, (loss_x, 3.08), (loss_w, 0.58), "Text-Text\nintra-modal", COLORS["loss"], dashed=True, fontsize=6.4)
    box(ax, (loss_x, 2.23), (loss_w, 0.58), "Hard negative\ncaptions", COLORS["loss"], dashed=True, fontsize=6.4)
    tag(ax, loss_x + 0.12, 2.10, "I2T direction only", color="0.28")
    box(ax, (loss_x, 1.34), (loss_w, 0.58), "Object\nclassification head", COLORS["loss"], dashed=True, fontsize=6.2)
    tag(ax, loss_x + 0.12, 1.21, "768-d input; COCO-only", color="0.28")

    elbow_arrow(ax, [(12.25, 5.72), (13.10, 5.72), (13.10, 5.96), (loss_x, 5.96)], color=MIXED_ARROW, lw=1.2)
    elbow_arrow(ax, [(12.25, 3.60), (13.36, 3.60), (13.36, 5.82), (loss_x, 5.82)], color=MIXED_ARROW, lw=1.2)
    arrow(ax, (loss_x + 0.74, 5.62), (loss_x + 0.74, 5.33), dashed=True, color=MIXED_ARROW)
    elbow_arrow(ax, [(12.25, 5.50), (12.70, 5.50), (12.70, 4.21), (loss_x, 4.21)], dashed=True, color=IMAGE_ARROW)
    elbow_arrow(ax, [(12.25, 3.38), (12.92, 3.38), (12.92, 3.37), (loss_x, 3.37)], dashed=True, color=TEXT_ARROW)
    elbow_arrow(ax, [(loss_x + loss_w, 5.92), (15.82, 5.92), (15.82, 2.52), (loss_x + loss_w, 2.52)], dashed=True, color=MIXED_ARROW)
    elbow_arrow(ax, [(12.25, 5.36), (16.02, 5.36), (16.02, 1.63), (loss_x + loss_w, 1.63)], dashed=True, color=IMAGE_ARROW)

    handles = [
        Patch(facecolor=COLORS["loss"], edgecolor=EDGE, label="Loss"),
        Patch(facecolor=COLORS["data"], edgecolor=EDGE, label="Data"),
        Patch(facecolor=COLORS["arch"], edgecolor=EDGE, label="Architectural"),
        Patch(facecolor=COLORS["segment"], edgecolor=EDGE, label="Segment-aware"),
        Patch(facecolor=COLORS["default"], edgecolor=EDGE, label="Default-frozen"),
        Line2D([0], [0], color=EDGE, linestyle="--", linewidth=1.0, label="dashed = optional"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.075), frameon=False, fontsize=6.4, ncol=6)
    ax.text(
        7.30,
        0.18,
        "Base-min = solid path only · Base-aug = solid + intra-modal dashed · other configurations toggle the labeled dashed components.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="0.30",
    )

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / "00_methodology_overview.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SAVE_FIG_DIR / "00_methodology_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
