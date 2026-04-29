"""
src/model_blip.py
-----------------
Diagnostic encoder ablation: CLIP ViT-L/14@336 image encoder + BLIP BERT-base
text encoder (bidirectional).

Hypothesis: CLIP's causal-masked (unidirectional) text attention contributes to
the COCO–Flickr representational gap. Bidirectional context via BLIP's BERT-base
text encoder may improve fine-grained compositional understanding.

This is NOT part of the B0–B5 experiment ladder. It is reported in a dedicated
"encoder bidirectionality" subsection.

Architecture:
    Image:  CLIP ViT-L/14@336 → visual_projection → 768-d (pretrained, partially unfrozen)
    Text:   BLIP BERT-base (ITM, unimodal mode — no cross-attention) → [CLS] → 768-d
            → text_projection (Linear 768→768, randomly initialized, trainable)
    Temp:   Reuses self.clip.logit_scale so that train.py / grad_cache.py references
            to self.model.clip.logit_scale work unchanged.

Usage:
    Activated by config:
        model.text_encoder: blip
        model.text_model_name: "Salesforce/blip-itm-base-coco"
    See registry.yaml BLIP_TEXT entry.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, BlipForImageTextRetrieval

logger = logging.getLogger(__name__)


class DualEncoderBLIPText(nn.Module):
    """
    Hybrid Dual Encoder: CLIP vision + BLIP text.

    API-compatible with DualEncoder — exposes the same forward(), encode_image(),
    encode_text(), and self.clip.logit_scale interface so the Trainer, loss module,
    GradCache, and eval paths work without modification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── 1. CLIP vision encoder ─────────────────────────────────────────
        clip_model_name = config['model']['image_model_name']
        logger.info(f"Loading CLIP vision encoder: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        self.clip = self.clip.float()
        self.clip_embed_dim = self.clip.config.projection_dim  # 768 for ViT-L/14
        logger.info(f"CLIP projection dimension: {self.clip_embed_dim}")

        # ── 2. BLIP text encoder (BERT-base, bidirectional) ────────────────
        blip_model_name = config['model']['text_model_name']
        logger.info(f"Loading BLIP text encoder: {blip_model_name}")
        blip_full = BlipForImageTextRetrieval.from_pretrained(
            blip_model_name, use_safetensors=True,
        )
        blip_full = blip_full.float()
        # Extract only the text encoder (BertModel with cross-attention layers).
        # When called WITHOUT encoder_hidden_states, the cross-attention layers
        # are skipped → pure bidirectional self-attention (unimodal mode).
        self.blip_text = blip_full.text_encoder
        # Free BLIP's vision encoder and ITM head — we only need the text side.
        del blip_full
        blip_hidden = self.blip_text.config.hidden_size  # 768 for BERT-base
        logger.info(f"BLIP text encoder hidden_size: {blip_hidden}")

        # ── 3. Text projection head (768 → 768) ───────────────────────────
        # BLIP's 768-d [CLS] output lives in a different embedding space than
        # CLIP's 768-d visual features. A trainable linear projection bridges
        # the two spaces. Randomly initialized (Xavier).
        self.text_projection = nn.Linear(blip_hidden, self.clip_embed_dim)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)
        logger.info(
            f"Text projection: Linear({blip_hidden}, {self.clip_embed_dim}) "
            "(randomly initialized, trainable)"
        )

        # ── 4. Learnable temperature ──────────────────────────────────────
        # Reuse self.clip.logit_scale so that train.py / grad_cache.py can
        # access self.model.clip.logit_scale without any code changes.
        # Reset to fine-tuning starting value (same as DualEncoder).
        with torch.no_grad():
            self.clip.logit_scale.fill_(2.6593)  # ln(1/0.07)
        logger.info("logit_scale reset to 2.6593 (τ=0.07)")

        # ── 5. Freezing strategy ──────────────────────────────────────────
        self._apply_freezing_strategy()

        # ── 6. Not a B5 variant — no segment injection / cls head ─────────
        self.seg_mode = None
        self.cls_head = None
        self.image_proj = None
        self.text_proj = None

    # ------------------------------------------------------------------ #
    #  Freezing
    # ------------------------------------------------------------------ #

    def _apply_freezing_strategy(self):
        """
        Freezing policy:
            CLIP:  Same as DualEncoder — freeze everything, unfreeze visual_projection
                   + last N vision blocks + logit_scale.
            BLIP:  Text encoder fully frozen (pretrained BERT weights).
                   text_projection is trainable (randomly initialized).
        """
        # ── CLIP: freeze all, selectively unfreeze ──
        for param in self.clip.parameters():
            param.requires_grad = False

        # Unfreeze CLIP visual_projection
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True

        # Unfreeze last N vision blocks
        num_vision_layers = self.config.get('model', {}).get('unfreeze_vision_layers', 0)
        unfreeze_strategy = self.config.get('model', {}).get('unfreeze_strategy', 'full')
        if num_vision_layers > 0:
            total_blocks = len(self.clip.vision_model.encoder.layers)
            unfreeze_blocks = list(range(total_blocks - num_vision_layers, total_blocks))
            logger.info(f"Unfreezing Vision Blocks: {unfreeze_blocks} (strategy: {unfreeze_strategy})")
            for block_idx in unfreeze_blocks:
                block = self.clip.vision_model.encoder.layers[block_idx]
                for name, param in block.named_parameters():
                    if self._should_unfreeze_param(name, unfreeze_strategy):
                        param.requires_grad = True
            if unfreeze_strategy in ['full', 'layernorm']:
                for param in self.clip.vision_model.post_layernorm.parameters():
                    param.requires_grad = True

        # Unfreeze logit_scale
        self.clip.logit_scale.requires_grad = True

        # ── BLIP text encoder: fully frozen ──
        for param in self.blip_text.parameters():
            param.requires_grad = False

        # ── text_projection: trainable (default requires_grad=True) ──

        self._print_freezing_summary()

    def _should_unfreeze_param(self, name, strategy):
        """Determine if a parameter should be unfrozen based on strategy.

        Mirrors DualEncoder._should_unfreeze_param exactly.
        """
        if strategy == "full":
            return True
        elif strategy == "attention":
            return any(kw in name for kw in ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj'])
        elif strategy == "mlp":
            return any(kw in name for kw in ['mlp', 'fc1', 'fc2'])
        elif strategy == "layernorm":
            return 'layer_norm' in name.lower() or 'layernorm' in name.lower()
        elif strategy == "bias":
            return 'bias' in name
        else:
            logger.warning(f"Unknown unfreeze strategy '{strategy}', defaulting to 'full'")
            return True

    def _print_freezing_summary(self):
        """Print trainable parameter summary."""
        clip_total = sum(p.numel() for p in self.clip.parameters())
        clip_trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        blip_total = sum(p.numel() for p in self.blip_text.parameters())
        blip_trainable = sum(p.numel() for p in self.blip_text.parameters() if p.requires_grad)
        proj_trainable = sum(p.numel() for p in self.text_projection.parameters() if p.requires_grad)

        logger.info(
            f"CLIP vision: {clip_trainable:,} trainable / {clip_total:,} total | "
            f"BLIP text: {blip_trainable:,} trainable / {blip_total:,} total (frozen) | "
            f"text_projection: {proj_trainable:,} trainable | "
            f"Total trainable: {clip_trainable + blip_trainable + proj_trainable:,}"
        )

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #

    def _get_image_features(self, images):
        """Extract image features via CLIP vision encoder (no seg_ids support)."""
        image_embeds = self.clip.get_image_features(pixel_values=images)
        return image_embeds.float()

    def _get_text_features(self, input_ids, attention_mask):
        """Extract text features via BLIP BERT encoder (unimodal, bidirectional).

        Uses the text encoder WITHOUT encoder_hidden_states, so the cross-attention
        layers are skipped. The [CLS] token output is projected to the CLIP joint space.
        """
        outputs = self.blip_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token is at position 0
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        text_embeds = self.text_projection(cls_output.float())  # [B, 768]
        return text_embeds

    # ------------------------------------------------------------------ #
    #  Public API (matches DualEncoder interface)
    # ------------------------------------------------------------------ #

    def forward(self, images, input_ids, attention_mask, seg_ids=None):
        """
        Forward pass returning L2-normalized embeddings.

        Args:
            images: [B, 3, 336, 336] pixel values
            input_ids: [B, SeqLen] BERT WordPiece token IDs
            attention_mask: [B, SeqLen] attention mask
            seg_ids: ignored (B5 not supported in this ablation)

        Returns:
            img_embeds: [B, 768] L2-normalized
            txt_embeds: [B, 768] L2-normalized
        """
        if seg_ids is not None:
            raise ValueError(
                "DualEncoderBLIPText does not support seg_ids (B5 variants). "
                "This is a diagnostic text-encoder ablation only."
            )
        image_embeds = self._get_image_features(images)
        text_embeds = self._get_text_features(input_ids, attention_mask)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        return image_embeds, text_embeds

    def encode_image(self, images, seg_ids=None):
        """Encode images only (L2-normalized)."""
        if seg_ids is not None:
            raise ValueError("DualEncoderBLIPText does not support seg_ids.")
        image_embeds = self._get_image_features(images)
        return F.normalize(image_embeds, p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        """Encode text only (L2-normalized)."""
        text_embeds = self._get_text_features(input_ids, attention_mask)
        return F.normalize(text_embeds, p=2, dim=1)
