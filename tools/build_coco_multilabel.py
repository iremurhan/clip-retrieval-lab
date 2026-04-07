"""
tools/build_coco_multilabel.py
------------------------------
Build a multi-label classification map from COCO instance annotations.

Parses instances_train2014.json and instances_val2014.json, maps each image_id
to a multi-hot tensor of shape (80,) over the 80 COCO categories.

Usage:
    python tools/build_coco_multilabel.py \
        --ann_dir ~/experiments/datasets/coco/annotations \
        --output datasets/coco/coco_multilabel_map.pt
"""

import argparse
import json
import os
import torch


# COCO has 80 categories but category IDs are not contiguous (1-90 with gaps).
# Build a deterministic mapping from category_id -> contiguous index [0, 79].
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
CAT_ID_TO_IDX = {cat_id: idx for idx, cat_id in enumerate(COCO_CAT_IDS)}


def parse_annotations(ann_path: str, label_map: dict[int, torch.Tensor]) -> int:
    """
    Parse a single COCO instances JSON and update label_map in-place.

    Args:
        ann_path: Path to instances_*.json
        label_map: Dict mapping image_id -> multi-hot tensor [80]

    Returns:
        Number of annotations processed.
    """
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Validate category IDs match expected COCO categories
    file_cat_ids = sorted(c['id'] for c in data['categories'])
    if file_cat_ids != sorted(COCO_CAT_IDS):
        raise ValueError(
            f"Category IDs in {ann_path} do not match expected 80 COCO categories. "
            f"Got {len(file_cat_ids)} categories."
        )

    count = 0
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        idx = CAT_ID_TO_IDX.get(cat_id)
        if idx is None:
            raise ValueError(
                f"Unknown category_id {cat_id} in {ann_path}. "
                f"Expected one of the 80 COCO categories."
            )

        if img_id not in label_map:
            label_map[img_id] = torch.zeros(80, dtype=torch.float32)
        label_map[img_id][idx] = 1.0
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build COCO multi-label map (image_id -> 80-dim multi-hot tensor)"
    )
    parser.add_argument(
        "--ann_dir", type=str, required=True,
        help="Directory containing instances_train2014.json and instances_val2014.json"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/coco/coco_multilabel_map.pt",
        help="Output path for the .pt file"
    )
    args = parser.parse_args()

    label_map: dict[int, torch.Tensor] = {}

    total = 0
    for split in ["train2014", "val2014"]:
        ann_path = os.path.join(args.ann_dir, f"instances_{split}.json")
        n = parse_annotations(ann_path, label_map)
        print(f"  {split}: {n} annotations processed")
        total += n

    print(f"Total: {total} annotations, {len(label_map)} unique images")

    # Sanity checks
    for img_id, vec in label_map.items():
        if vec.sum() == 0:
            raise RuntimeError(
                f"Image {img_id} has zero labels after parsing. "
                "This should not happen — every image should have at least one annotation."
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(label_map, args.output)
    print(f"Saved label map to {args.output}")


if __name__ == "__main__":
    main()
