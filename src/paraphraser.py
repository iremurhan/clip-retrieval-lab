"""
src/paraphraser.py
------------------
Two paraphraser implementations for intra-modal text contrastive loss
L_text_text = InfoNCE(T_orig, T_paraphrased):

  NLTKParaphraser      — fast, CPU-only: synonym replacement via WordNet.
                         No GPU, no LLM, no external API. Default path.

  PrecomputedParaphraser — tokenizes paraphrase strings already embedded
                           in the dataset JSON (generated offline). Fastest.
"""

import random
import logging

import torch
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


class NLTKParaphraser:
    """
    Simple synonym replacement paraphraser using NLTK WordNet.
    Replaces ~replacement_prob fraction of eligible words (nouns, verbs,
    adjectives) with a random WordNet synonym. CPU-only, no LLM required.

    Args:
        tokenizer:         CLIPTokenizer — tokenizes paraphrases for encode_text().
        device:            torch.device — output tensors placed on this device.
        max_length:        int — CLIP token sequence length (default: 77).
        replacement_prob:  float — probability of replacing each eligible word.
        seed:              int — RNG seed for reproducibility.
    """

    _POS_MAP = {
        'NN': wordnet.NOUN, 'NNS': wordnet.NOUN,
        'VB': wordnet.VERB, 'VBD': wordnet.VERB,
        'VBG': wordnet.VERB, 'VBN': wordnet.VERB,
        'JJ': wordnet.ADJ,  'JJR': wordnet.ADJ,  'JJS': wordnet.ADJ,
    }

    def __init__(self, tokenizer, device: torch.device, max_length: int = 77,
                 replacement_prob: float = 0.3, seed: int = 42):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.replacement_prob = replacement_prob
        self.rng = random.Random(seed)

    def _get_synonym(self, word: str, pos: str) -> str:
        """Return a random WordNet synonym; falls back to original word if none found."""
        synsets = wordnet.synsets(word, pos=pos)
        synonyms = []
        for synset in synsets:
            for lemma in synset.lemmas():
                candidate = lemma.name().replace('_', ' ')
                if candidate.lower() != word.lower():
                    synonyms.append(candidate)
        if not synonyms:
            return word
        return self.rng.choice(synonyms)

    def _paraphrase_caption(self, caption: str) -> str:
        """Replace a random subset of nouns/verbs/adjectives with WordNet synonyms."""
        tokens = nltk.word_tokenize(caption)
        tagged = nltk.pos_tag(tokens)
        result = []
        for word, tag in tagged:
            wn_pos = self._POS_MAP.get(tag)
            if wn_pos and self.rng.random() < self.replacement_prob:
                result.append(self._get_synonym(word, wn_pos))
            else:
                result.append(word)
        return ' '.join(result)

    def generate(self, captions: list) -> tuple:
        """
        Generate paraphrases for a batch of captions and tokenize for CLIP.

        Args:
            captions: list[str] of length N

        Returns:
            para_input_ids:      [N, max_length] LongTensor on self.device
            para_attention_mask: [N, max_length] LongTensor on self.device
        """
        paraphrases = [self._paraphrase_caption(c) for c in captions]
        tokenized = self.tokenizer(
            paraphrases,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return (
            tokenized['input_ids'].to(self.device),       # [N, max_length]
            tokenized['attention_mask'].to(self.device),  # [N, max_length]
        )


class PrecomputedParaphraser:
    """
    Tokenizes pre-loaded paraphrase strings that were embedded in the dataset
    JSON by scripts/precompute_paraphrases.py.

    No LLM is loaded. Training speed is unaffected.

    Args:
        tokenizer:   CLIPTokenizer — used to tokenize paraphrase strings.
        device:      torch.device — output tensors placed on this device.
        max_length:  int — token sequence length (default: 77).
    """

    def __init__(self, tokenizer, device: torch.device, max_length: int = 77):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def generate(self, paraphrases: list) -> tuple:
        """
        Tokenizes pre-loaded paraphrase strings for CLIP encode_text().

        Falls back to empty string if a paraphrase is None or empty. Logs a
        warning when more than 10% of the batch has empty paraphrases.

        Args:
            paraphrases: list[str] of length N — paraphrase strings from batch

        Returns:
            para_input_ids:      [N, max_length] LongTensor on self.device
            para_attention_mask: [N, max_length] LongTensor on self.device
        """
        texts = [p if p else "" for p in paraphrases]

        empty_count = sum(1 for t in texts if t == "")
        if empty_count / len(texts) > 0.1:
            logger.warning(
                f"PrecomputedParaphraser: {empty_count}/{len(texts)} samples "
                f"({100 * empty_count / len(texts):.1f}%) have empty paraphrases. "
                f"Intra-text loss contribution will be near zero for these samples."
            )

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
