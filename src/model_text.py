"""
Generic text-tower control: CLIP ViT-L/14@336 image encoder + frozen
Hugging Face transformer text encoder.

This is the non-COCO-tuned counterpart to BLIP_TEXT. It keeps the same shape
as the BLIP text-side diagnostic: frozen bidirectional text tower, [CLS] token,
trainable Linear(hidden_size -> CLIP projection dim), and CLIP logit_scale.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel

logger = logging.getLogger(__name__)


class DualEncoderTransformerText(nn.Module):
    """Hybrid dual encoder with CLIP vision and a frozen text-only transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        clip_model_name = config["model"]["image_model_name"]
        logger.info(f"Loading CLIP vision encoder: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        self.clip = self.clip.float()
        self.clip_embed_dim = self.clip.config.projection_dim

        text_model_name = config["model"]["text_model_name"]
        logger.info(f"Loading frozen transformer text encoder: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_encoder = self.text_encoder.float()
        text_hidden = self.text_encoder.config.hidden_size

        self.text_projection = nn.Linear(text_hidden, self.clip_embed_dim)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

        with torch.no_grad():
            self.clip.logit_scale.fill_(2.6593)

        self._apply_freezing_strategy()
        self.seg_mode = None
        self.cls_head = None
        self.image_proj = None
        self.text_proj = None

    def _apply_freezing_strategy(self):
        for param in self.clip.parameters():
            param.requires_grad = False

        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True

        num_vision_layers = self.config.get("model", {}).get("unfreeze_vision_layers", 0)
        unfreeze_strategy = self.config.get("model", {}).get("unfreeze_strategy", "full")
        if num_vision_layers > 0:
            total_blocks = len(self.clip.vision_model.encoder.layers)
            unfreeze_blocks = list(range(total_blocks - num_vision_layers, total_blocks))
            logger.info(f"Unfreezing Vision Blocks: {unfreeze_blocks} (strategy: {unfreeze_strategy})")
            for block_idx in unfreeze_blocks:
                block = self.clip.vision_model.encoder.layers[block_idx]
                for name, param in block.named_parameters():
                    if self._should_unfreeze_param(name, unfreeze_strategy):
                        param.requires_grad = True
            if unfreeze_strategy in ["full", "layernorm"]:
                for param in self.clip.vision_model.post_layernorm.parameters():
                    param.requires_grad = True

        self.clip.logit_scale.requires_grad = True

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self._print_freezing_summary()

    def _should_unfreeze_param(self, name, strategy):
        if strategy == "full":
            return True
        if strategy == "attention":
            return any(kw in name for kw in ["self_attn", "q_proj", "k_proj", "v_proj", "out_proj"])
        if strategy == "mlp":
            return any(kw in name for kw in ["mlp", "fc1", "fc2"])
        if strategy == "layernorm":
            return "layer_norm" in name.lower() or "layernorm" in name.lower()
        if strategy == "bias":
            return "bias" in name
        logger.warning(f"Unknown unfreeze strategy '{strategy}', defaulting to 'full'")
        return True

    def _print_freezing_summary(self):
        clip_total = sum(p.numel() for p in self.clip.parameters())
        clip_trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        text_total = sum(p.numel() for p in self.text_encoder.parameters())
        text_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        proj_trainable = sum(p.numel() for p in self.text_projection.parameters() if p.requires_grad)
        logger.info(
            f"CLIP vision: {clip_trainable:,} trainable / {clip_total:,} total | "
            f"text tower: {text_trainable:,} trainable / {text_total:,} total (frozen) | "
            f"text_projection: {proj_trainable:,} trainable | "
            f"Total trainable: {clip_trainable + text_trainable + proj_trainable:,}"
        )

    def _get_image_features(self, images):
        return self.clip.get_image_features(pixel_values=images).float()

    def _get_text_features(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
        return self.text_projection(cls_output.float())

    def forward(self, images, input_ids, attention_mask, seg_ids=None, sam_features=None):
        if seg_ids is not None:
            raise ValueError("DualEncoderTransformerText does not support seg_ids.")
        if sam_features is not None:
            raise ValueError("DualEncoderTransformerText does not support sam_features.")
        image_embeds = self._get_image_features(images)
        text_embeds = self._get_text_features(input_ids, attention_mask)
        return F.normalize(image_embeds, p=2, dim=1), F.normalize(text_embeds, p=2, dim=1)

    def encode_image(self, images, seg_ids=None, sam_features=None):
        if seg_ids is not None:
            raise ValueError("DualEncoderTransformerText does not support seg_ids.")
        if sam_features is not None:
            raise ValueError("DualEncoderTransformerText does not support sam_features.")
        return F.normalize(self._get_image_features(images), p=2, dim=1)

    def encode_image_patches(self, images):
        vision_out = self.clip.vision_model(pixel_values=images)
        patch_tokens = vision_out.last_hidden_state[:, 1:, :].float()
        patch_tokens = self.clip.vision_model.post_layernorm(patch_tokens)
        bsz, num_patches, hidden = patch_tokens.shape
        patch_embeds = self.clip.visual_projection(patch_tokens.reshape(bsz * num_patches, hidden))
        patch_embeds = patch_embeds.reshape(bsz, num_patches, -1)
        return F.normalize(patch_embeds, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self._get_text_features(input_ids, attention_mask), p=2, dim=1)
