"""
src/paraphraser.py
------------------
LaCLIP-style paraphraser for intra-modal text contrastive loss
L_text_text = InfoNCE(T_orig, T_paraphrased).

Loads LLM-generated caption rewrites from a JSON file produced offline by
scripts/generate_rewrites.py and returns a random rewrite per sample.
JSON format: {sentid_str: [rewrite1, rewrite2, ...]}.
"""

import json
import os
import random
import logging

import torch

logger = logging.getLogger(__name__)


class PrecomputedLLMParaphraser:
    """
    Loads LLM-generated caption rewrites and returns a random rewrite per sentid.

    Args:
        rewrites_path: str — path to the caption_rewrites.json file.
        tokenizer:     CLIPTokenizer — tokenizes selected rewrites.
        device:        torch.device — output tensors placed on this device.
        max_length:    int — CLIP token sequence length (default: 77).
        seed:          int — RNG seed for reproducibility.
    """

    def __init__(self, rewrites_path: str, tokenizer, device: torch.device,
                 max_length: int = 77, seed: int = 42):
        if not rewrites_path or not os.path.exists(rewrites_path):
            raise FileNotFoundError(
                f"LLM rewrites file not found: {rewrites_path}. "
                "Run scripts/generate_rewrites.py first."
            )
        with open(rewrites_path, 'r') as f:
            raw = json.load(f)
        self.rewrites = {int(k): v for k, v in raw.items()}
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.rng = random.Random(seed)
        logger.info(
            f"PrecomputedLLMParaphraser loaded {len(self.rewrites)} sentids "
            f"from {rewrites_path}"
        )

    def generate(self, sentids: list) -> tuple:
        """
        Pick a random rewrite for each sentid and tokenize for CLIP.

        Args:
            sentids: list[int] of length N — sentence IDs from the batch.

        Returns:
            para_input_ids:      [N, max_length] LongTensor on self.device
            para_attention_mask: [N, max_length] LongTensor on self.device
        """
        texts = []
        for sid in sentids:
            sid_int = int(sid)
            rw_list = self.rewrites.get(sid_int)
            if not rw_list:
                raise KeyError(
                    f"sentid {sid_int} not found in rewrites file. "
                    f"Rewrites file is incomplete — re-run scripts/generate_rewrites.py."
                )
            texts.append(self.rng.choice(rw_list))

        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return (
            tokenized['input_ids'].to(self.device),       # [N, max_length]
            tokenized['attention_mask'].to(self.device),  # [N, max_length]
        )

    def sample_pair(self, sentids: list) -> tuple:
        """
        Sample TWO distinct rewrites per sentid (without replacement) from the
        offline-precomputed JSON, for the paraphrase ↔ paraphrase intra-modal
        text loss. No runtime LLM generation — selection only.

        Each sentid must have at least 2 rewrites available; otherwise a ValueError
        is raised so the caller fails loud rather than silently sampling the same
        paraphrase twice.

        Args:
            sentids: list[int] of length N — sentence IDs from the batch.

        Returns:
            (a_input_ids, a_attention_mask, b_input_ids, b_attention_mask) —
            each [N, max_length] LongTensor on self.device.
        """
        texts_a = []
        texts_b = []
        for sid in sentids:
            sid_int = int(sid)
            if sid_int not in self.rewrites:
                raise KeyError(
                    f"sentid {sid_int} not found in rewrites file. It was "
                    f"excluded by the offline rewrite script (paraphrase "
                    f"collapse — < num_rewrites unique candidates). Run a "
                    f"retry pass over the .failed.jsonl log to recover."
                )
            rw_list = self.rewrites[sid_int]
            n = len(rw_list)
            if n < 2:
                raise ValueError(
                    f"sentid {sid_int}: need >= 2 precomputed rewrites for the "
                    f"paraphrase pair, got {n}. Re-run the offline rewrite script "
                    "with --num_rewrites>=2."
                )
            a, b = self.rng.sample(rw_list, 2)  # 2 distinct rewrites
            texts_a.append(a)
            texts_b.append(b)

        tok_a = self.tokenizer(
            texts_a, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        tok_b = self.tokenizer(
            texts_b, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        return (
            tok_a['input_ids'].to(self.device),
            tok_a['attention_mask'].to(self.device),
            tok_b['input_ids'].to(self.device),
            tok_b['attention_mask'].to(self.device),
        )
