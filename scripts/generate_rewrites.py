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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    logger.info(f"Model loaded: {model_name}")
    return model, tokenizer


def generate_for_caption(
    model, tokenizer, caption: str, num_candidates: int,
) -> list[str]:
    """Generate multiple rewrite candidates for a single caption.

    Returns list of decoded candidate strings (may contain duplicates).
    Raises torch.cuda.OutOfMemoryError on CUDA OOM (caller handles retry).
    """
    messages = build_chat_messages(caption)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]  # [1, seq_len]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )  # [num_candidates, seq_len + new_tokens]

    candidates = []
    for seq in outputs:
        decoded = tokenizer.decode(
            seq[prompt_len:], skip_special_tokens=True,  # [new_tokens]
        ).strip()
        # Take only the first line — the prompt asks for a single caption
        first_line = decoded.split("\n")[0].strip()
        if first_line:
            candidates.append(first_line)

    return candidates


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
    captions: list[dict],
    output_path: str,
    model_name: str,
    num_rewrites: int = 3,
    checkpoint_every: int = 200,
    resume: bool = False,
) -> None:
    """Main generation loop with per-caption inference, checkpointing, and resume.

    Note: inference is per-caption (not batched across captions). The
    ``checkpoint_every`` argument controls how frequently progress is
    persisted to disk, measured in number of processed captions.
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

    for i, item in enumerate(remaining):
        sentid = item["sentid"]
        caption = item["caption"]

        success = False
        for attempt in range(2):  # at most 2 attempts (initial + 1 OOM retry)
            try:
                candidates = generate_for_caption(
                    model, tokenizer, caption, NUM_CANDIDATES,
                )
                rewrites = select_unique_rewrites(
                    candidates, caption, num_rewrites,
                )
                all_rewrites[sentid] = rewrites
                success = True
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 0:
                    logger.warning(
                        f"CUDA OOM for sentid {sentid}, retrying after cache clear..."
                    )
                else:
                    logger.error(
                        f"CUDA OOM for sentid {sentid} on retry. "
                        f"Skipping (resume will catch it later)."
                    )

        if not success:
            skipped_sentids.append(sentid)

        # Periodic checkpoint
        if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
            _save(all_rewrites, output_path)
            logger.info(
                f"[{i+1}/{total}] Saved {len(all_rewrites)} sentids to {output_path}"
            )

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
        "--checkpoint_every", type=int, default=200,
        help="Persist progress every N captions (default: 200)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file",
    )
    args = parser.parse_args()

    captions = load_captions(args.captions_path)
    generate_rewrites(
        captions,
        output_path=args.output_path,
        model_name=args.model,
        num_rewrites=args.num_rewrites,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
