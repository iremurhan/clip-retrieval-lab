"""
tools/fold_recovered_rewrites.py
--------------------------------
Fold a recovered-rewrites JSON (produced by tools/retry_failed_rewrites.py)
into the canonical caption_rewrites_<model>.json. Atomic dict update + atomic
file replace. Optionally rewrites the .failed.jsonl to drop now-recovered
sentids so it reflects the residual state.

Pre-flight:
- Verifies recovered sentids do NOT already exist in canonical (would mean
  the canonical was already updated; aborts to avoid silent overwrite).

Usage:
    python tools/fold_recovered_rewrites.py \\
        --canonical datasets/<ds>/caption_rewrites_<model_slug>.json \\
        --recovered datasets/<ds>/caption_rewrites_<model_slug>.recovered.json \\
        [--failed datasets/<ds>/caption_rewrites_<model_slug>.failed.jsonl]
"""

import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--canonical", type=str, required=True,
                   help="caption_rewrites_<model>.json (will be updated in place atomically)")
    p.add_argument("--recovered", type=str, required=True,
                   help="recovered.json from retry_failed_rewrites.py")
    p.add_argument("--failed", type=str, default="",
                   help="Optional: .failed.jsonl whose now-recovered sentids will be removed")
    args = p.parse_args()

    for path in (args.canonical, args.recovered):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: file not found: {path}")

    with open(args.canonical, "r") as f:
        canonical_raw = json.load(f)
    canonical = {int(k): v for k, v in canonical_raw.items()}

    with open(args.recovered, "r") as f:
        recovered_raw = json.load(f)
    recovered = {int(k): v for k, v in recovered_raw.items()}

    print(f"Canonical: {len(canonical):,} sentids ({args.canonical})")
    print(f"Recovered: {len(recovered):,} sentids ({args.recovered})")

    if not recovered:
        print("Nothing to fold.")
        return

    # Pre-flight: any recovered sentid already present in canonical means we
    # would silently overwrite. The retry pipeline derives failed.jsonl from
    # canonical-MISSING sentids (or K<2 entries), so an overlap is a bug.
    overlap = set(recovered) & set(canonical)
    if overlap:
        sample = list(overlap)[:5]
        sys.exit(
            f"ERROR: {len(overlap)} recovered sentids already exist in "
            f"canonical (sample: {sample}). Refusing to overwrite. Inspect "
            f"the recovered file or run the retry pipeline against a fresh "
            f"canonical."
        )

    canonical.update(recovered)

    tmp = args.canonical + ".tmp"
    with open(tmp, "w") as f:
        json.dump({str(k): v for k, v in canonical.items()}, f)
    os.replace(tmp, args.canonical)
    print(f"Folded {len(recovered):,} sentids -> {args.canonical} "
          f"(new total: {len(canonical):,})")

    # Optionally rewrite the failed.jsonl to drop now-recovered sentids.
    if args.failed and os.path.isfile(args.failed):
        kept = []
        dropped = 0
        with open(args.failed, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if int(rec["sentid"]) in recovered:
                    dropped += 1
                else:
                    kept.append(line)
        tmp_f = args.failed + ".tmp"
        with open(tmp_f, "w") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp_f, args.failed)
        print(f"Rewrote {args.failed}: kept {len(kept):,}, dropped {dropped:,} "
              "(now-recovered)")


if __name__ == "__main__":
    main()
