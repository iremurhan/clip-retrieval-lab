"""
tools/retry_failed_rewrites.py
------------------------------
Retry K=1 (or K=0) paraphrase failures from a previous run with elevated
sampling on the SAME model. Reads a JSONL of failed records (produced by the
main generate_rewrites.py path's *.failed.jsonl, or by
check_rewrite_coverage.py --dump_failed for legacy data), regenerates fresh
candidates per caption with elevated temperature/top_k, combines with the
existing partial rewrite(s), deduplicates, and writes:

    <recovered_path>     JSON dict {sentid -> [rewrite_a, rewrite_b]}
                         for sentids that reached K >= num_rewrites.
    <still_failed_path>  JSONL of sentids that remain K < num_rewrites
                         even after retry (truly hard cases).

The recovered.json is then folded into the canonical caption_rewrites_<model>.json
by tools/fold_recovered_rewrites.py.

Usage:
    python tools/retry_failed_rewrites.py \\
        --failed_path datasets/<ds>/caption_rewrites_<model_slug>.failed.jsonl \\
        --recovered_path datasets/<ds>/caption_rewrites_<model_slug>.recovered.json \\
        --still_failed_path datasets/<ds>/caption_rewrites_<model_slug>.still_failed.jsonl \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --num_rewrites 2
"""

import argparse
import json
import logging
import os
import sys
import time

# Make legacy.paraphrase_generation.generate_rewrites importable so we can
# reuse load_model, generate_for_batch, and select_unique_rewrites without
# duplicating the inference plumbing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GENGEN = os.path.normpath(os.path.join(_HERE, "..", "legacy", "paraphrase_generation"))
sys.path.insert(0, _GENGEN)

from generate_rewrites import (  # noqa: E402
    load_model,
    generate_for_batch,
    select_unique_rewrites,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# Retry sampling: more aggressive than the main pass to widen the candidate
# distribution where the main pass collapsed. T=1.0 with top_k=100 lets the
# tail of the distribution contribute, NUM_CANDIDATES_RETRY=16 over-generates
# so dedup has more material to find a 2nd unique form.
NUM_CANDIDATES_RETRY = 16
RETRY_TEMPERATURE = 1.0
RETRY_TOP_P = 0.95
RETRY_TOP_K = 100


def load_failed_records(path: str) -> list:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _save_recovered(recovered: dict, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        # Stringify keys for stable JSON output (paraphraser converts to int on load)
        json.dump({str(k): v for k, v in recovered.items()}, f)
    os.replace(tmp, path)


def _save_still_failed(records: list, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--failed_path", type=str, required=True)
    p.add_argument("--recovered_path", type=str, required=True)
    p.add_argument("--still_failed_path", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num_rewrites", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=200)
    args = p.parse_args()

    if not os.path.isfile(args.failed_path):
        logger.error(f"Failed path not found: {args.failed_path}")
        sys.exit(1)

    records = load_failed_records(args.failed_path)
    if not records:
        logger.info("No records to retry. Exiting.")
        _save_recovered({}, args.recovered_path)
        _save_still_failed([], args.still_failed_path)
        return

    logger.info(
        f"Loaded {len(records)} failed records from {args.failed_path}. "
        f"Retry sampling: T={RETRY_TEMPERATURE}, top_p={RETRY_TOP_P}, "
        f"top_k={RETRY_TOP_K}, num_candidates={NUM_CANDIDATES_RETRY}."
    )

    model, tokenizer = load_model(args.model)

    recovered: dict = {}
    still_failed: list = []
    total = len(records)
    processed_since_save = 0
    t_start = time.time()

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch = records[batch_start:batch_end]
        captions = [r["caption"] for r in batch]
        partials = [list(r.get("rewrites", [])) for r in batch]

        cand_lists = generate_for_batch(
            model, tokenizer, captions, NUM_CANDIDATES_RETRY,
            temperature=RETRY_TEMPERATURE,
            top_p=RETRY_TOP_P,
            top_k=RETRY_TOP_K,
        )

        for r, partial, cands in zip(batch, partials, cand_lists):
            # partial rewrites first so they get priority during dedup; fresh
            # candidates only fill gaps. select_unique_rewrites uses the same
            # normalized dedup as the main pass.
            combined = partial + cands
            rewrites = select_unique_rewrites(combined, args.num_rewrites)
            sid = int(r["sentid"])
            if len(rewrites) >= args.num_rewrites:
                recovered[sid] = rewrites
            else:
                still_failed.append({
                    "sentid": sid,
                    "caption": r["caption"],
                    "n_unique": len(rewrites),
                    "rewrites": rewrites,
                })

        processed = batch_end
        processed_since_save += (batch_end - batch_start)
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0.0
        eta_min = (total - processed) / rate / 60 if rate > 0 else float("inf")
        logger.info(
            f"[{processed}/{total}] {rate:.2f} caps/s, ETA {eta_min:.1f} min, "
            f"recovered={len(recovered)} still_failed={len(still_failed)}"
        )

        if processed_since_save >= args.checkpoint_every or processed == total:
            _save_recovered(recovered, args.recovered_path)
            _save_still_failed(still_failed, args.still_failed_path)
            processed_since_save = 0
            logger.info(
                f"Checkpoint: recovered={len(recovered)}, still_failed={len(still_failed)} "
                f"-> {args.recovered_path}, {args.still_failed_path}"
            )

    # Final flush (no-op if last loop iteration already saved)
    _save_recovered(recovered, args.recovered_path)
    _save_still_failed(still_failed, args.still_failed_path)
    recovery_pct = 100.0 * len(recovered) / total if total else 0.0
    logger.info(
        f"Done. Recovered {len(recovered)}/{total} sentids "
        f"({recovery_pct:.1f}%). {len(still_failed)} still failed."
    )


if __name__ == "__main__":
    main()
