"""
Build caption-complexity buckets for Karpathy test captions.

Outputs one CSV per dataset under the external figure artifact cache with token
and concept quartiles.
The concept proxy is spaCy noun chunks plus adjective tokens, using
en_core_web_sm exactly so the diagnostic is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/Volumes/T7/Research/experiments/datasets")
ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
FIG_ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_FIG_ARTIFACT_ROOT", ARTIFACT_ROOT / "figs")
)
DEFAULT_OUTPUT_DIR = Path(os.environ.get("CAPTION_BUCKET_OUTPUT_DIR", FIG_ARTIFACT_ROOT / "cache"))
DATASETS = ("coco", "flickr30k")


def captions_path(dataset: str, data_root: Path) -> Path:
    if dataset == "coco":
        return data_root / "coco" / "caption_datasets" / "dataset_coco.json"
    if dataset == "flickr30k":
        return data_root / "flickr30k" / "caption_datasets" / "dataset_flickr30k.json"
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_test_captions(dataset: str, data_root: Path) -> pd.DataFrame:
    path = captions_path(dataset, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Karpathy captions JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    for image in payload["images"]:
        split = "train" if image.get("split") == "restval" else image.get("split")
        if split != "test":
            continue
        if "cocoid" in image:
            image_id = int(image["cocoid"])
        elif "imgid" in image:
            image_id = int(image["imgid"])
        elif "id" in image:
            image_id = int(image["id"])
        else:
            raise ValueError(f"Image entry missing id field: {image.keys()}")

        for sent in image["sentences"][:5]:
            rows.append(
                {
                    "caption_id": int(sent["sentid"]),
                    "image_id": image_id,
                    "caption_text": str(sent["raw"]).strip(),
                }
            )
    if not rows:
        raise ValueError(f"No test captions found in {path}")
    return pd.DataFrame(rows)


def load_spacy_model():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required. Install it with:\n"
            "  .venv-eval/bin/python -m spacy download en_core_web_sm"
        ) from exc


def add_complexity_columns(df: pd.DataFrame, nlp) -> pd.DataFrame:
    out = df.copy()
    out["token_count"] = out["caption_text"].map(lambda text: len(str(text).split()))

    noun_counts: list[int] = []
    adj_counts: list[int] = []
    texts = out["caption_text"].astype(str).tolist()
    for doc in nlp.pipe(texts, batch_size=256):
        noun_counts.append(sum(1 for _ in doc.noun_chunks))
        adj_counts.append(sum(1 for token in doc if token.pos_ == "ADJ"))

    out["noun_chunk_count"] = noun_counts
    out["adj_count"] = adj_counts
    out["concept_count"] = out["noun_chunk_count"] + out["adj_count"]
    return out


def add_rank_quartile(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    # Rank first so repeated integer counts still produce four nearly equal
    # buckets. Boundaries are reported as the observed min/max per bucket.
    labels = ["Q1", "Q2", "Q3", "Q4"]
    ranks = df[source_col].rank(method="first")
    df[target_col] = pd.qcut(ranks, q=4, labels=labels)
    return df


def print_bucket_report(dataset: str, df: pd.DataFrame) -> None:
    print(f"\n{dataset}")
    for source_col, bucket_col in [
        ("token_count", "token_quartile"),
        ("concept_count", "concept_quartile"),
    ]:
        quantiles = df[source_col].quantile([0.25, 0.5, 0.75]).to_dict()
        print(
            f"  {source_col} quartile cut values: "
            f"Q25={quantiles[0.25]:.2f}, Q50={quantiles[0.5]:.2f}, Q75={quantiles[0.75]:.2f}"
        )
        counts = df[bucket_col].value_counts().sort_index()
        ranges = df.groupby(bucket_col, observed=True)[source_col].agg(["min", "max"])
        print(f"  {bucket_col} counts/ranges:")
        for bucket in ["Q1", "Q2", "Q3", "Q4"]:
            count = int(counts.get(bucket, 0))
            lo = int(ranges.loc[bucket, "min"]) if bucket in ranges.index else -1
            hi = int(ranges.loc[bucket, "max"]) if bucket in ranges.index else -1
            print(f"    {bucket}: n={count} range=[{lo}, {hi}]")

        expected = len(df) / 4.0
        max_dev = (counts.astype(float).sub(expected).abs().max() / expected) if expected else 0.0
        if max_dev > 0.10:
            print(f"  WARNING: {bucket_col} sizes differ by more than +/-10%.")


def build_dataset(dataset: str, data_root: Path, output_dir: Path, nlp, force: bool) -> Path:
    output_path = output_dir / f"caption_buckets_{dataset}.csv"
    if output_path.exists() and not force:
        df = pd.read_csv(output_path)
        print_bucket_report(dataset, df)
        print(f"  cache exists: {output_path}")
        return output_path

    df = load_test_captions(dataset, data_root)
    df = add_complexity_columns(df, nlp)
    df = add_rank_quartile(df, "token_count", "token_quartile")
    df = add_rank_quartile(df, "concept_count", "concept_quartile")

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "caption_id",
        "image_id",
        "caption_text",
        "token_count",
        "noun_chunk_count",
        "adj_count",
        "concept_count",
        "token_quartile",
        "concept_quartile",
    ]
    df[columns].to_csv(output_path, index=False)
    print_bucket_report(dataset, df)
    print(f"  wrote: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build caption complexity buckets for Karpathy test captions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    nlp = load_spacy_model()
    for dataset in args.datasets:
        build_dataset(dataset, args.data_root, args.output_dir, nlp, args.force)


if __name__ == "__main__":
    main()
