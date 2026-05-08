"""
tools/clean_canonical_rewrites.py
---------------------------------
Normalize a legacy caption_rewrites_<model>.json that was produced by an old
generate_rewrites.py path which padded the rewrites list with the original
caption when fewer than num_rewrites unique candidates remained. Such padded
entries cause sample_pair() to return (orig, rewrite) pairs with non-trivial
probability — a leak into the intra-modal text loss.

This tool:
  1. For each sentid in the canonical, normalizes each rewrite string and
     drops entries equal to the original caption (under the same dedup
     normalization the patched generator uses).
  2. If K (unique non-original rewrites) >= num_rewrites: writes the cleaned
     rewrites back to the canonical.
  3. If K < num_rewrites: removes the sentid from the canonical (so
     paraphraser.sample_pair raises KeyError instead of returning a leaked
     pair) and emits a record into a .failed.jsonl that the retry tool can
     consume.

Atomic writes via tmp+rename.

Usage:
    python tools/clean_canonical_rewrites.py \\
        --karpathy datasets/flickr30k/caption_datasets/dataset_flickr30k.json \\
        --canonical datasets/flickr30k/caption_rewrites_<model_slug>.json \\
        --failed_out datasets/flickr30k/caption_rewrites_<model_slug>.failed.jsonl \\
        [--num_rewrites 2]
"""

import argparse
import json
import os
import re
import sys


_DEDUP_TRAIL_PUNCT = " .!?,;:"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(_DEDUP_TRAIL_PUNCT)


def load_originals(karpathy_path: str) -> dict:
    with open(karpathy_path, "r") as f:
        data = json.load(f)
    orig: dict = {}
    for img in data["images"]:
        if img["split"] not in ("train", "restval"):
            continue
        for sent in img["sentences"][:5]:
            orig[int(sent["sentid"])] = sent["raw"]
    return orig


def clean_one(rewrites: list, original: str) -> list:
    """Drop empty/original-equal/duplicate rewrites under normalized dedup.
    Returns the kept-in-order unique non-original rewrites."""
    orig_key = norm(original)
    seen = set()
    kept = []
    for r in rewrites:
        rk = norm(r)
        if not rk or rk == orig_key or rk in seen:
            continue
        seen.add(rk)
        kept.append(r.strip())
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--karpathy", type=str, required=True)
    p.add_argument("--canonical", type=str, required=True)
    p.add_argument("--failed_out", type=str, required=True)
    p.add_argument("--num_rewrites", type=int, default=2)
    args = p.parse_args()

    for path in (args.karpathy, args.canonical):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: file not found: {path}")

    orig_by_sid = load_originals(args.karpathy)
    with open(args.canonical, "r") as f:
        raw = json.load(f)
    canonical = {int(k): v for k, v in raw.items()}

    print(f"Karpathy sentids:  {len(orig_by_sid):,}")
    print(f"Canonical sentids: {len(canonical):,}")

    cleaned: dict = {}
    failed: list = []
    sentids_with_padded = 0
    sentids_unchanged = 0

    for sid, rw_list in canonical.items():
        original = orig_by_sid.get(sid)
        if original is None:
            # Dead entry not in karpathy: keep as-is, can't normalize without orig.
            cleaned[sid] = rw_list
            continue
        kept = clean_one(rw_list, original)
        if len(kept) == len(rw_list):
            sentids_unchanged += 1
        else:
            sentids_with_padded += 1
        if len(kept) >= args.num_rewrites:
            cleaned[sid] = kept
        else:
            failed.append({
                "sentid": int(sid),
                "caption": original,
                "n_unique": len(kept),
                "rewrites": kept,
            })

    print()
    print(f"Sentids with padded entries removed: {sentids_with_padded:,}")
    print(f"Sentids unchanged:                   {sentids_unchanged:,}")
    print(f"Sentids retained in canonical (K>={args.num_rewrites}): {len(cleaned):,}")
    print(f"Sentids dropped to failed (K<{args.num_rewrites}):       {len(failed):,}")

    # Atomic-write cleaned canonical
    tmp_c = args.canonical + ".tmp"
    with open(tmp_c, "w") as f:
        json.dump({str(k): v for k, v in cleaned.items()}, f)
    os.replace(tmp_c, args.canonical)
    print(f"\nWrote cleaned canonical -> {args.canonical}")

    # Atomic-write failed.jsonl
    tmp_f = args.failed_out + ".tmp"
    os.makedirs(os.path.dirname(tmp_f) or ".", exist_ok=True)
    with open(tmp_f, "w") as f:
        for rec in failed:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp_f, args.failed_out)
    print(f"Wrote failed records   -> {args.failed_out}")


if __name__ == "__main__":
    main()
