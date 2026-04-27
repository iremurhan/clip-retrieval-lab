"""
scripts/generate_rewrites.py
-----------------------------
Offline caption rewriting using a local LLM via HuggingFace Transformers.

For each caption in the Karpathy JSON (train + restval splits), generates
N diverse rewrites via a local instruction-tuned LLM. Output JSON:
{sentid: [rewrite1, ..., rewriteN]}.

Uses AutoModelForCausalLM with bfloat16 and device_map="auto".

Supports --resume to continue from a partial output file.

Usage (GPU required — submit via sbatch):
    python scripts/generate_rewrites.py \
        --captions_path datasets/coco/caption_datasets/dataset_coco.json \
        --output_path datasets/coco/caption_rewrites.json \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --num_rewrites 3 \
        --checkpoint_every 200 \
        --resume

    python scripts/generate_rewrites.py \
        --captions_path datasets/flickr30k/caption_datasets/dataset_flickr30k.json \
        --output_path datasets/flickr30k/caption_rewrites.json \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --num_rewrites 3 \
        --resume
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force line-buffered stdout so SLURM logs flush immediately
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

SYSTEM_PROMPT = (
    "You are a caption rewriting assistant. Rewrite image captions with "
    "different sentence structure while preserving ALL objects, attributes, "
    "relationships, and actions. Do not add or remove any visual information. "
    "Output only the rewritten caption, nothing else."
)

NUM_CANDIDATES = 6  # generate this many candidates, then pick the most distinct


# ============================================================================
# Data loading
# ============================================================================

def load_captions(captions_path: str) -> list[dict]:
    """Load train + restval captions from Karpathy JSON.

    Returns list of {"sentid": int, "caption": str}.
    """
    with open(captions_path, "r") as f:
        data = json.load(f)

    captions = []
    for img in data["images"]:
        split = img["split"]
        if split not in ("train", "restval"):
            continue
        for sent in img["sentences"][:5]:
            captions.append({
                "sentid": int(sent["sentid"]),
                "caption": sent["raw"],
            })

    logger.info(f"Loaded {len(captions)} captions from {captions_path}")
    return captions


# ============================================================================
# Prompt building & rewrite selection
# ============================================================================

def build_chat_messages(caption: str) -> list[dict]:
    """Build chat messages for a single caption.

    Prepends the system prompt into the user message because several
    instruction-tuned models (e.g. Mistral-7B-Instruct-v0.2) do not
    support a dedicated system role in their chat template.
    """
    return [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\nRewrite this caption:\n{caption}",
        },
    ]


def select_unique_rewrites(
    candidates: list[str], original: str, num_rewrites: int,
) -> list[str]:
    """Pick the most distinct rewrites from candidates.

    Deduplicates by lowercased text. If fewer than num_rewrites unique
    candidates remain, pads with whatever was found, then fills remaining
    slots with the original caption.
    """
    seen = set()
    unique = []
    for c in candidates:
        c_stripped = c.strip()
        if not c_stripped:
            continue
        key = c_stripped.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c_stripped)

    if len(unique) < num_rewrites:
        logger.warning(
            f"Only {len(unique)}/{num_rewrites} unique rewrites for: "
            f"{original[:80]!r}. Padding with original."
        )
        while len(unique) < num_rewrites:
            unique.append(original)

    return unique[:num_rewrites]


# ============================================================================
# HuggingFace Transformers inference
# ============================================================================

def load_model(model_name: str) -> tuple:
    """Load model and tokenizer via HuggingFace Transformers."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only LMs require LEFT padding so generated tokens align across the batch
    tokenizer.padding_side = "left"

    logger.info(f"Loading model weights (bfloat16, device_map=auto): {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def generate_for_batch(
    model, tokenizer, captions: list, num_candidates: int,
) -> list:
    """Generate rewrite candidates for a BATCH of captions in a single forward pass.

    Args:
        model:          loaded HF causal LM
        tokenizer:      tokenizer with padding_side='left'
        captions:       list[str] of length B
        num_candidates: number of candidates per caption (num_return_sequences)

    Returns:
        list of length B, each entry is a list[str] of candidate rewrites for
        that caption (after stripping prompt + first-line extraction).

    Raises torch.cuda.OutOfMemoryError on CUDA OOM (caller handles retry).
    """
    prompts = [
        tokenizer.apply_chat_template(
            build_chat_messages(c), tokenize=False, add_generation_prompt=True,
        )
        for c in captions
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]  # [B, prompt_len] after left-padding

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )  # [B * num_candidates, prompt_len + new_tokens]

    # Slice off prompt portion (uniform thanks to left-padding)
    new_tokens = outputs[:, prompt_len:]  # [B * num_candidates, new_tokens]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Group num_candidates outputs per input caption
    batch_candidates = []
    for i in range(len(captions)):
        start = i * num_candidates
        end = start + num_candidates
        cands = []
        for text in decoded[start:end]:
            first_line = text.strip().split("\n")[0].strip()
            if first_line:
                cands.append(first_line)
        batch_candidates.append(cands)

    return batch_candidates


# ============================================================================
# Main generation loop
# ============================================================================

def _save(rewrites: dict, path: str) -> None:
    """Atomic write: write to tmp then rename."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(rewrites, f)
    os.replace(tmp, path)


def generate_rewrites(
    captions: list,
    output_path: str,
    model_name: str,
    num_rewrites: int = 3,
    batch_size: int = 8,
    checkpoint_every: int = 200,
    resume: bool = False,
) -> None:
    """Main generation loop with batched inference, checkpointing, and resume.

    Inference is batched across captions: ``batch_size`` prompts go through
    one ``.generate()`` call, producing ``batch_size * NUM_CANDIDATES`` output
    sequences. ``checkpoint_every`` is measured in number of processed
    captions (not batches), and is rounded up to the nearest batch boundary.
    """

    # Load existing progress if resuming
    existing = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            raw = json.load(f)
        existing = {int(k): v for k, v in raw.items()}
        logger.info(f"Resuming: {len(existing)} sentids already processed")

    remaining = [c for c in captions if c["sentid"] not in existing]
    logger.info(f"{len(remaining)} captions to process ({len(existing)} done)")

    if not remaining:
        logger.info("All captions already processed. Nothing to do.")
        return

    model, tokenizer = load_model(model_name)

    all_rewrites = dict(existing)
    skipped_sentids = []
    total = len(remaining)
    processed_since_save = 0
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = remaining[batch_start:batch_end]
        batch_captions = [it["caption"] for it in batch_items]
        batch_sentids = [it["sentid"] for it in batch_items]

        # Try batched generation; on OOM, halve the batch and retry, falling
        # back to per-caption inference at batch_size=1.
        success = False
        current_bs = len(batch_captions)
        sub_batches = [(batch_captions, batch_sentids)]
        for attempt in range(3):
            try:
                for sub_caps, sub_sids in sub_batches:
                    cand_lists = generate_for_batch(
                        model, tokenizer, sub_caps, NUM_CANDIDATES,
                    )
                    for sid, cap, cands in zip(sub_sids, sub_caps, cand_lists):
                        all_rewrites[sid] = select_unique_rewrites(
                            cands, cap, num_rewrites,
                        )
                success = True
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                logger.warning(
                    f"CUDA OOM at batch {batch_start}-{batch_end}; "
                    f"retrying with sub-batch size {current_bs}"
                )
                # Re-chunk into smaller pieces
                sub_batches = []
                for i in range(0, len(batch_captions), current_bs):
                    sub_batches.append((
                        batch_captions[i:i + current_bs],
                        batch_sentids[i:i + current_bs],
                    ))

        if not success:
            logger.error(
                f"Batch {batch_start}-{batch_end} failed permanently. "
                f"Skipping {len(batch_sentids)} sentids."
            )
            skipped_sentids.extend(batch_sentids)

        processed = batch_end
        processed_since_save += (batch_end - batch_start)

        # Throughput log every batch
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0.0
        eta_min = (total - processed) / rate / 60 if rate > 0 else float("inf")
        logger.info(
            f"[{processed}/{total}] {rate:.2f} caps/s, ETA {eta_min:.1f} min"
        )

        # Periodic checkpoint
        if processed_since_save >= checkpoint_every or processed == total:
            _save(all_rewrites, output_path)
            processed_since_save = 0
            logger.info(f"Checkpoint: {len(all_rewrites)} sentids → {output_path}")

    _save(all_rewrites, output_path)
    logger.info(f"Done. {len(all_rewrites)} sentids written to {output_path}")
    if skipped_sentids:
        logger.warning(
            f"{len(skipped_sentids)} sentids skipped due to errors. "
            f"Re-run with --resume to retry them."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM caption rewrites (LaCLIP-style) using HuggingFace Transformers"
    )
    parser.add_argument(
        "--captions_path", type=str, required=True,
        help="Path to Karpathy JSON (e.g. datasets/coco/caption_datasets/dataset_coco.json)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output JSON path (e.g. datasets/coco/caption_rewrites.json)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num_rewrites", type=int, default=3,
        help="Number of rewrites per caption (default: 3)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Captions per .generate() call (default: 8). Total seqs per call = batch_size * NUM_CANDIDATES.",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=200,
        help="Persist progress every N captions (default: 200)",
    )
    parser.add_argument(
        "--smoke_test", type=int, default=0,
        help="If >0, only process the first N captions (for quick validation).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file",
    )
    args = parser.parse_args()

    captions = load_captions(args.captions_path)
    if args.smoke_test > 0:
        captions = captions[:args.smoke_test]
        logger.info(f"SMOKE TEST mode: limiting to first {len(captions)} captions")

    generate_rewrites(
        captions,
        output_path=args.output_path,
        model_name=args.model,
        num_rewrites=args.num_rewrites,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
