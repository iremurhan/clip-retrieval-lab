"""
src/paraphraser.py
------------------
On-the-fly text paraphrasing using Mistral-7B-Instruct-v0.2 (4-bit NF4 quantized).

Generates semantic positive pairs (T_orig, T_paraphrased) for the intra-modal
text contrastive loss L_text_text = InfoNCE(T_orig, T_paraphrased).

The LLM is permanently frozen (eval mode, no gradients) and runs entirely under
torch.no_grad(). Output paraphrases are re-tokenized with the CLIP tokenizer
before being routed to encode_text().
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class OnTheFlyParaphraser:
    """
    Batched on-the-fly paraphrasing using Mistral-7B-Instruct-v0.2.

    Loads the LLM in 4-bit NF4 quantization (bitsandbytes). The model is
    permanently frozen and never receives gradients.

    Args:
        para_config (dict): config['paraphraser'] section, must contain:
            - model_name: HuggingFace model ID
            - max_new_tokens: int
            - generation_temperature: float
            - do_sample: bool
            - bnb_4bit_quant_type: str (e.g. 'nf4')
            - bnb_4bit_use_double_quant: bool
        clip_tokenizer: CLIPTokenizer — used to re-tokenize decoded paraphrases
        device: torch.device — LLM placed on this device
    """

    def __init__(self, para_config: dict, clip_tokenizer, device: torch.device):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=para_config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=para_config['bnb_4bit_use_double_quant'],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Try flash attention 2 for reduced activation peak; fall back silently
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                para_config['model_name'],
                quantization_config=bnb_config,
                device_map={"": device},
                attn_implementation='flash_attention_2',
            )
        except (ValueError, ImportError):
            self.llm = AutoModelForCausalLM.from_pretrained(
                para_config['model_name'],
                quantization_config=bnb_config,
                device_map={"": device},
            )

        self.llm.eval()
        self.llm.requires_grad_(False)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(para_config['model_name'])
        # Left-padding is required for batched causal (decoder-only) generation
        self.llm_tokenizer.padding_side = 'left'
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.clip_tokenizer = clip_tokenizer
        self.max_new_tokens = para_config['max_new_tokens']
        self.temperature = para_config['generation_temperature']
        self.do_sample = para_config['do_sample']
        self.device = device

    def generate(self, captions: list) -> tuple:
        """
        Generate paraphrases for a batch of captions and re-tokenize for CLIP.

        Args:
            captions: list[str] of length N — raw caption strings from the batch

        Returns:
            para_input_ids:      [N, 77] LongTensor on self.device
            para_attention_mask: [N, 77] LongTensor on self.device
        """
        prompts = [self._build_prompt(c) for c in captions]

        llm_enc = self.llm_tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)
        # llm_enc.input_ids: [N, P]  (left-padded)

        with torch.no_grad():
            generated_ids = self.llm.generate(
                input_ids=llm_enc['input_ids'],
                attention_mask=llm_enc['attention_mask'],
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        # generated_ids: [N, P + generated_len]

        # Slice off the prompt prefix — only decode the newly generated tokens
        prompt_len = llm_enc['input_ids'].shape[1]
        new_tokens = generated_ids[:, prompt_len:]  # [N, generated_len]
        paraphrases = self.llm_tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        # paraphrases: list[str], len=N

        # Re-tokenize with CLIP tokenizer (max_length=77, padded)
        clip_enc = self.clip_tokenizer(
            paraphrases,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt',
        )
        para_input_ids = clip_enc['input_ids'].to(self.device)            # [N, 77]
        para_attention_mask = clip_enc['attention_mask'].to(self.device)  # [N, 77]

        return para_input_ids, para_attention_mask

    @staticmethod
    def _build_prompt(caption: str) -> str:
        """Mistral-7B-Instruct-v0.2 instruction format: [INST] ... [/INST]"""
        return (
            "[INST] Paraphrase the following image caption using different words. "
            "Preserve the meaning. Output only the paraphrase, nothing else.\n"
            f"{caption} [/INST]"
        )
