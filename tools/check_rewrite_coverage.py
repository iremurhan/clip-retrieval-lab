"""
tools/check_rewrite_coverage.py
-------------------------------
Audit a caption_rewrites JSON for paraphrase-pair coverage under the current
normalized-dedup semantics (the same logic used by
legacy/paraphrase_generation/generate_rewrites.py:select_unique_rewrites).

For each sentid, counts unique non-original rewrites K after normalizing:
    norm(s) = re.sub(r"\\s+", " ", s.strip().lower()).rstrip(" .!?,;:")

Threshold: paraphraser.sample_pair needs K >= 2.

Reports:
- K-distribution histogram
- Failure rate at K<2 (the operational fail-fast threshold)
- Failure rate at K<3 (the legacy num_rewrites=3 threshold, for cross-checks)
- A sample of failing sentids for spot inspection
- Optional: dumps a .failed_audit.jsonl with full records for downstream retry

Usage:
    python tools/check_rewrite_coverage.py \\
        --karpathy datasets/flickr30k/caption_datasets/dataset_flickr30k.json \\
        --rewrites datasets/flickr30k/caption_rewrites_meta-llama-3-8b-instruct.json

    # Optional: write per-failure JSONL for the retry pass
    python tools/check_rewrite_coverage.py \\
        --karpathy ... --rewrites ... \\
        --dump_failed datasets/flickr30k/caption_rewrites_<model>.audit.failed.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter


_DEDUP_TRAIL_PUNCT = " .!?,;:"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(_DEDUP_TRAIL_PUNCT)


def load_originals(karpathy_path: str) -> dict:
    """Build sentid -> original caption from the Karpathy split JSON.

    Uses train + restval splits (matches generate_rewrites.load_captions).
    """
    with open(karpathy_path, "r") as f:
        data = json.load(f)
    orig: dict = {}
    for img in data["images"]:
        if img["split"] not in ("train", "restval"):
            continue
        for sent in img["sentences"][:5]:
            orig[int(sent["sentid"])] = sent["raw"]
    return orig


def k_for_sentid(rewrites: list, original: str) -> tuple:
    """Return (K, unique_rewrites) where K is the count of unique non-original
    rewrites under normalized dedup."""
    orig_key = norm(original)
    seen: set = set()
    unique: list = []
    for r in rewrites:
        r_key = norm(r)
        if not r_key or r_key == orig_key or r_key in seen:
            continue
        seen.add(r_key)
        unique.append(r)
    return len(unique), unique


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--karpathy", type=str, required=True,
                        help="Karpathy split JSON path.")
    parser.add_argument("--rewrites", type=str, required=True,
                        help="caption_rewrites_<model>.json path.")
    parser.add_argument("--dump_failed", type=str, default="",
                        help="Optional JSONL path to dump failed records "
                             "(K<2) for downstream retry.")
    parser.add_argument("--sample_n", type=int, default=5,
                        help="Number of failing sentids to print for inspection.")
    args = parser.parse_args()

    orig_by_sid = load_originals(args.karpathy)
    with open(args.rewrites, "r") as f:
        raw = json.load(f)
    rewrites_by_sid = {int(k): v for k, v in raw.items()}

    print(f"Karpathy train+restval sentids:  {len(orig_by_sid):>9,}")
    print(f"Rewrites JSON sentids:           {len(rewrites_by_sid):>9,}")

    missing_in_rewrites = set(orig_by_sid) - set(rewrites_by_sid)
    extra_in_rewrites = set(rewrites_by_sid) - set(orig_by_sid)
    print(f"In karpathy but not in rewrites: {len(missing_in_rewrites):>9,}  "
          "(would raise KeyError in paraphraser)")
    print(f"In rewrites but not in karpathy: {len(extra_in_rewrites):>9,}  "
          "(harmless, just dead entries)")

    intersect = set(orig_by_sid) & set(rewrites_by_sid)
    k_counter: Counter = Counter()
    failed_records: list = []
    sample_failures: list = []
    for sid in intersect:
        orig = orig_by_sid[sid]
        rw_list = rewrites_by_sid[sid]
        k, unique = k_for_sentid(rw_list, orig)
        k_counter[k] += 1
        if k < 2:
            failed_records.append({
                "sentid": int(sid),
                "caption": orig,
                "n_unique": k,
                "rewrites": unique,
            })
            if len(sample_failures) < args.sample_n:
                sample_failures.append((sid, orig, k, unique))

    n = sum(k_counter.values())
    print(f"\nK-distribution over {n:,} intersected sentids:")
    for k in sorted(k_counter.keys()):
        c = k_counter[k]
        print(f"  K={k}: {c:>9,}  ({100*c/n:.3f}%)")

    fail_at_3 = sum(c for k, c in k_counter.items() if k < 3)
    fail_at_2 = sum(c for k, c in k_counter.items() if k < 2)
    fail_at_1 = sum(c for k, c in k_counter.items() if k < 1)
    print(f"\nFailure rate @ K<3 (legacy threshold): {fail_at_3:,}  "
          f"({100*fail_at_3/n:.3f}%)")
    print(f"Failure rate @ K<2 (current threshold): {fail_at_2:,}  "
          f"({100*fail_at_2/n:.3f}%)")
    print(f"Failure rate @ K<1 (totally collapsed): {fail_at_1:,}  "
          f"({100*fail_at_1/n:.3f}%)")

    if missing_in_rewrites:
        print(f"\nNote: {len(missing_in_rewrites):,} karpathy sentids are not "
              f"present in the rewrites JSON at all. paraphraser.sample_pair "
              f"will raise KeyError on those.")

    if sample_failures:
        print(f"\nSample of K<2 sentids (showing {len(sample_failures)}):")
        for sid, orig, k, unique in sample_failures:
            print(f"  sentid={sid}  K={k}  caption={orig[:70]!r}")
            for r in unique:
                print(f"      rewrite: {r[:70]!r}")

    if args.dump_failed:
        with open(args.dump_failed, "w") as f:
            for rec in failed_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nDumped {len(failed_records):,} failed records to {args.dump_failed}")


if __name__ == "__main__":
    main()
