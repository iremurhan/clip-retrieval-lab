from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "cache" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from helpers import SAVE_FIG_DIR, setup_thesis_style


THUMBNAIL_PATH = Path("/Volumes/T7/Research/experiments/datasets/coco/val2014/COCO_val2014_000000391895.jpg")
CAPTION_TEXT = "A man in a red shirt and a red hat is on a motorcycle on a hill side."

COLORS = {
    "input": "#F7F7F7",
    "image": "#D9EAF7",
    "text": "#E4DDF2",
    "encoder": "#E6E6E6",
    "embedding": "#FFF2CC",
    "loss": "#BFD8EA",
    "nograd": "#EFEFEF",
}
EDGE = "#2F2F2F"
IMAGE_EDGE = "#2C7FB8"
TEXT_EDGE = "#7A5195"
LOSS_EDGE = "#08519C"
STOP = "#7A7A7A"


def box(ax, xy, wh, label, fc, ec=EDGE, dashed=False, fontsize=8, lw=1.0, zorder=2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, color=EDGE, dashed=False, lw=1.0, rad=0.0, zorder=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stop_slash(ax, x, y, size=0.12, color=STOP):
    ax.plot([x - size, x + size], [y - size, y + size], color=color, linewidth=1.4, zorder=7)
    ax.plot([x - size * 0.75, x + size * 1.25], [y - size * 1.25, y + size * 0.75], color="white", linewidth=0.8, zorder=8)


def thumbnail(ax, xy, wh, label, variant="clean"):
    x, y = xy
    w, h = wh
    patch = box(ax, xy, wh, "", COLORS["input"], ec=IMAGE_EDGE, fontsize=6)
    image_extent = (x + 0.06, x + w - 0.06, y + 0.18, y + h - 0.06)
    if THUMBNAIL_PATH.exists():
        img = mpimg.imread(THUMBNAIL_PATH)
        if variant == "aug_a":
            img = img[: int(img.shape[0] * 0.86), int(img.shape[1] * 0.06) :, :]
        elif variant == "aug_b":
            img = img[int(img.shape[0] * 0.08) :, : int(img.shape[1] * 0.90), :]
            img = (img.astype("float32") * [0.92, 1.04, 1.05]).clip(0, 255).astype(img.dtype)
        ax.imshow(img, extent=image_extent, aspect="auto", clip_path=patch, clip_on=True, zorder=3)
    else:
        box(ax, (x + 0.06, y + 0.18), (w - 0.12, h - 0.24), "", "#D0D0D0", ec="#D0D0D0", zorder=3)
    ax.text(x + w / 2, y + 0.08, label, ha="center", va="center", fontsize=6.1, family="monospace", zorder=4)
    return patch


def caption_box(ax, xy, wh, label, text, fontsize=6.2):
    x, y = xy
    w, h = wh
    box(ax, xy, wh, "", COLORS["input"], ec=TEXT_EDGE, fontsize=6)
    ax.text(x + 0.07, y + h - 0.09, text, ha="left", va="top", fontsize=fontsize, zorder=3)
    ax.text(x + w / 2, y + 0.07, label, ha="center", va="center", fontsize=5.9, family="monospace", zorder=4)


def no_grad_wrap(ax, xy, wh):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor=COLORS["nograd"],
        edgecolor="#777777",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.08, y + h - 0.08, "no_grad", ha="left", va="top", fontsize=6.2, color="#555555", zorder=5)
    return patch


def encoder(ax, xy, text, edge_color):
    return box(ax, xy, (1.40, 0.52), text, COLORS["encoder"], ec=edge_color, fontsize=7.1, lw=1.1)


def embedding(ax, xy, label, edge_color):
    return box(ax, xy, (0.88, 0.42), label, COLORS["embedding"], ec=edge_color, fontsize=7.2, lw=1.0)


def loss_box(ax, xy, wh, title, symbol):
    x, y = xy
    w, h = wh
    box(ax, xy, wh, title, COLORS["loss"], ec=LOSS_EDGE, fontsize=7.1, lw=1.1)
    ax.text(x + w / 2, y + 0.12, symbol, ha="center", va="center", fontsize=8.3, color=LOSS_EDGE, zorder=4)


def main() -> None:
    setup_thesis_style()
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.30, 7.05, "Intra-modal augmentation losses.", fontsize=10.6, weight="bold", va="top")
    ax.text(0.30, 6.66, "Clean image-text alignment plus symmetric image-image and text-text consistency.", fontsize=7.2, color="0.30")

    # Row labels
    ax.text(0.24, 5.62, "clean pair", fontsize=6.6, weight="bold", color="0.25", va="center")
    ax.text(0.24, 3.56, "augmented\nviews\n(image)", fontsize=6.4, weight="bold", color="0.25", va="center", linespacing=1.05)
    ax.text(0.24, 1.54, "paraphrases\n(text)", fontsize=6.4, weight="bold", color="0.25", va="center", linespacing=1.05)

    # Row 1: inter-modal clean path.
    thumbnail(ax, (1.58, 5.36), (1.15, 0.78), "clean $x_i$", "clean")
    caption_box(ax, (1.58, 4.60), (1.15, 0.60), "clean $c_i$", "red-shirt rider\non hillside", fontsize=5.6)
    encoder(ax, (3.05, 5.47), "CLIP image\nencoder", IMAGE_EDGE)
    encoder(ax, (3.05, 4.64), "CLIP text\nencoder", TEXT_EDGE)
    embedding(ax, (5.06, 5.52), "$v_i$", IMAGE_EDGE)
    embedding(ax, (5.06, 4.69), "$t_i$", TEXT_EDGE)
    loss_box(ax, (7.10, 4.98), (1.92, 0.82), "Image-Text\nContrastive InfoNCE", r"$\mathcal{L}_{\mathrm{inter}}$")
    arrow(ax, (2.73, 5.75), (3.05, 5.74), IMAGE_EDGE)
    arrow(ax, (2.73, 4.90), (3.05, 4.90), TEXT_EDGE)
    arrow(ax, (4.45, 5.74), (5.06, 5.74), IMAGE_EDGE)
    arrow(ax, (4.45, 4.90), (5.06, 4.90), TEXT_EDGE)
    arrow(ax, (5.94, 5.73), (7.10, 5.54), IMAGE_EDGE)
    arrow(ax, (5.94, 4.90), (7.10, 5.24), TEXT_EDGE)

    # Row 2: image intra-modal path.
    thumbnail(ax, (1.58, 3.90), (1.15, 0.72), "aug a", "aug_a")
    no_grad_wrap(ax, (1.46, 2.96), (1.40, 0.88))
    thumbnail(ax, (1.58, 3.06), (1.15, 0.72), "aug b", "aug_b")
    encoder(ax, (3.05, 3.98), "CLIP image\nencoder", IMAGE_EDGE)
    encoder(ax, (3.05, 3.14), "CLIP image\nencoder", IMAGE_EDGE)
    ax.text(3.75, 3.78, "shared weights", ha="center", va="center", fontsize=5.9, color="0.35")
    ax.plot([3.75, 3.75], [3.66, 3.98], color="0.55", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    embedding(ax, (5.06, 4.03), r"$\tilde{v}_i^{(a)}$", IMAGE_EDGE)
    embedding(ax, (5.06, 3.19), r"$\tilde{v}_i^{(b)}$", IMAGE_EDGE)
    loss_box(ax, (7.10, 3.40), (1.92, 0.74), "Symmetric InfoNCE", r"$\mathcal{L}_{\mathrm{intra}}^{I}$")
    arrow(ax, (2.73, 4.26), (3.05, 4.24), IMAGE_EDGE)
    arrow(ax, (2.73, 3.42), (3.05, 3.40), IMAGE_EDGE)
    arrow(ax, (4.45, 4.24), (5.06, 4.24), IMAGE_EDGE)
    arrow(ax, (4.45, 3.40), (5.06, 3.40), IMAGE_EDGE)
    arrow(ax, (5.94, 4.24), (7.10, 3.93), IMAGE_EDGE, lw=1.1)
    ax.text(6.46, 4.26, r"$a\!\to\!b$", fontsize=6.4, color=IMAGE_EDGE)
    arrow(ax, (5.94, 3.40), (7.10, 3.61), IMAGE_EDGE, lw=1.1)
    stop_slash(ax, 6.45, 3.52)
    ax.text(6.46, 3.28, r"$b\!\to\!a$", fontsize=6.4, color=IMAGE_EDGE)

    # Row 3: text intra-modal path.
    caption_box(ax, (1.46, 1.92), (1.38, 0.56), "para a", "A red-shirted\nrider climbs a hill.", fontsize=5.0)
    no_grad_wrap(ax, (1.34, 0.96), (1.62, 0.88))
    caption_box(ax, (1.46, 1.06), (1.38, 0.56), "para b", "A rider in red\nis on a hillside.", fontsize=5.0)
    encoder(ax, (3.05, 1.96), "CLIP text\nencoder", TEXT_EDGE)
    encoder(ax, (3.05, 1.10), "CLIP text\nencoder", TEXT_EDGE)
    ax.text(3.75, 1.76, "shared weights", ha="center", va="center", fontsize=5.9, color="0.35")
    ax.plot([3.75, 3.75], [1.64, 1.96], color="0.55", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    embedding(ax, (5.06, 2.01), r"$\tilde{t}_i^{(a)}$", TEXT_EDGE)
    embedding(ax, (5.06, 1.15), r"$\tilde{t}_i^{(b)}$", TEXT_EDGE)
    loss_box(ax, (7.10, 1.36), (1.92, 0.74), "Symmetric InfoNCE", r"$\mathcal{L}_{\mathrm{intra}}^{T}$")
    arrow(ax, (2.84, 2.20), (3.05, 2.22), TEXT_EDGE)
    arrow(ax, (2.84, 1.34), (3.05, 1.36), TEXT_EDGE)
    arrow(ax, (4.45, 2.22), (5.06, 2.22), TEXT_EDGE)
    arrow(ax, (4.45, 1.36), (5.06, 1.36), TEXT_EDGE)
    arrow(ax, (5.94, 2.22), (7.10, 1.89), TEXT_EDGE, lw=1.1)
    ax.text(6.46, 2.24, r"$a\!\to\!b$", fontsize=6.4, color=TEXT_EDGE)
    arrow(ax, (5.94, 1.36), (7.10, 1.57), TEXT_EDGE, lw=1.1)
    stop_slash(ax, 6.45, 1.48)
    ax.text(6.46, 1.24, r"$b\!\to\!a$", fontsize=6.4, color=TEXT_EDGE)

    # Weight-sharing cue across rows.
    ax.text(4.20, 0.42, "Encoder boxes with the same label share weights across clean and augmented branches.", fontsize=6.8, color="0.35")

    fig.savefig(SAVE_FIG_DIR / "10_intra_modal_flow.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(SAVE_FIG_DIR / "10_intra_modal_flow.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
