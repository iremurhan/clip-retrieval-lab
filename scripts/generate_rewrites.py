"""
scripts/generate_rewrites.py
-----------------------------
Offline caption rewriting using a local LLM via vllm (LaCLIP-style).

For each caption in the Karpathy JSON (train + restval splits), generates
N diverse rewrites via a local instruction-tuned LLM. Output JSON:
{sentid: [rewrite1, ..., rewriteN]}.

Requires vllm (installed in the Docker image).

Supports --resume to continue from a partial output file.

Usage (GPU required — submit via sbatch):
    python scripts/generate_rewrites.py \
        --captions_path datasets/coco/caption_datasets/dataset_coco.json \
        --output_path datasets/coco/caption_rewrites.json \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --num_rewrites 3 \
        --batch_size 16 \
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
import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

SYSTEM_PROMPT = (
    "You are a caption rewriting assistant. You will receive a batch of "
    "numbered image captions. For each caption, produce exactly {n} diverse "
    "rewrites that use different sentence structures while preserving ALL "
    "objects, attributes, relationships, and actions. Do not add or remove "
    "any visual information.\n\n"
    "Output ONLY a valid JSON object — no markdown fences, no commentary, "
    "no text before or after the JSON. Format:\n"
    '{{"results": [{{"id": 1, "rewrites": ["rewrite1", ...]}}, '
    '{{"id": 2, "rewrites": ["rewrite1", ...]}}]}}'
)

USER_TEMPLATE = (
    "Rewrite each of the {count} captions below exactly {n} times. "
    "Vary sentence structure (passive voice, fronted adverbials, relative "
    "clauses, etc.). Output ONLY valid JSON.\n\n"
    "{numbered_captions}"
)


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
# Prompt building & response parsing
# ============================================================================

def build_chat_messages(batch: list[dict], num_rewrites: int) -> list[dict]:
    """Build chat messages (system + user) for a batch of captions."""
    numbered = "\n".join(
        f"{i+1}. {item['caption']}" for i, item in enumerate(batch)
    )
    system = SYSTEM_PROMPT.format(n=num_rewrites)
    user = USER_TEMPLATE.format(
        n=num_rewrites, count=len(batch), numbered_captions=numbered,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_response(response_text: str, batch: list[dict], num_rewrites: int) -> dict:
    """Parse LLM JSON response into {sentid: [rewrites]} dict.

    Raises ValueError on malformed output.
    """
    text = response_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]

    # Extract the outermost JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in response")
    text = json_match.group()

    parsed = json.loads(text)
    results = parsed["results"]

    if len(results) != len(batch):
        raise ValueError(
            f"Expected {len(batch)} results, got {len(results)}"
        )

    rewrites_map = {}
    for i, entry in enumerate(results):
        sentid = batch[i]["sentid"]
        rw = entry["rewrites"]
        if len(rw) != num_rewrites:
            raise ValueError(
                f"sentid {sentid}: expected {num_rewrites} rewrites, got {len(rw)}"
            )
        rewrites_map[sentid] = rw

    return rewrites_map


# ============================================================================
# vllm inference
# ============================================================================

def load_model(model_name: str) -> tuple:
    """Load model via vllm and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    logger.info(f"vllm model loaded: {model_name}")
    return llm, tokenizer


def generate_batch(
    llm: LLM, tokenizer, batch: list[dict], num_rewrites: int,
) -> str:
    """Generate rewrites for a batch of captions."""
    messages = build_chat_messages(batch, num_rewrites)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # max_tokens scales with batch size: ~80 tokens per caption × num_rewrites
    max_tokens = min(8192, len(batch) * num_rewrites * 80 + 256)

    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


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
    batch_size: int = 16,
    resume: bool = False,
    max_retries: int = 3,
) -> None:
    """Main generation loop with batching, checkpointing, and resume support."""

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

    llm, tokenizer = load_model(model_name)

    all_rewrites = dict(existing)
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    skipped_sentids = []

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining))
        batch = remaining[start:end]

        for attempt in range(1, max_retries + 1):
            try:
                response_text = generate_batch(llm, tokenizer, batch, num_rewrites)
                batch_rewrites = parse_response(response_text, batch, num_rewrites)
                all_rewrites.update(batch_rewrites)
                break
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(
                    f"Batch {batch_idx+1}/{total_batches} attempt {attempt}: "
                    f"parse error: {e}"
                )
                if attempt == max_retries:
                    batch_sids = [c['sentid'] for c in batch]
                    skipped_sentids.extend(batch_sids)
                    logger.error(
                        f"Batch {batch_idx+1} failed after {max_retries} attempts. "
                        f"Skipping sentids: {batch_sids}"
                    )

        # Checkpoint every 50 batches
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            _save(all_rewrites, output_path)
            logger.info(
                f"[{batch_idx+1}/{total_batches}] "
                f"Saved {len(all_rewrites)} sentids to {output_path}"
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
        description="Generate LLM caption rewrites (LaCLIP-style) using local vllm"
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
        "--batch_size", type=int, default=16,
        help="Captions per LLM call (default: 16)",
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
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
