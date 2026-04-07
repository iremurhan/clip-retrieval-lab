FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL maintainer="Irem Urhan"

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# B5_seg variant: bake the SAM ViT-B checkpoint into the image so that
# scripts/precompute_sam.py is self-contained on the SLURM worker.
# Source: https://github.com/facebookresearch/segment-anything (ViT-B, ~375 MB).
# If you prefer not to bloat the image, delete this RUN, host the checkpoint
# under /users/beyza.urhan/experiments/checkpoints/, and add a container-mount
# in scripts/train.slurm + scripts/precompute_sam.slurm.
RUN mkdir -p /weights && \
    wget -q -O /weights/sam_vit_b_01ec64.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && \
    test -s /weights/sam_vit_b_01ec64.pth