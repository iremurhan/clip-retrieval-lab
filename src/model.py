"""
src/model.py
------------
Model architecture for Cross-Modal Retrieval.
Implements DualEncoder using OpenAI CLIP as the backbone with optional projection heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import CLIPModel

logger = logging.getLogger(__name__)


class DualEncoder(nn.Module):
    """
    Dual Encoder using OpenAI CLIP (clip-vit-large-patch14-336).
    
    CLIP is pre-trained on 400M image-text pairs from the internet,
    providing extremely strong visual-semantic representations.
    Uses 336x336 input resolution for higher quality features.
    
    Strategy:
    - Freeze CLIP backbone (vision + text encoders)
    - Train only projection layers for domain adaptation
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config  # Store for use in freezing strategy
        self.dropout_p = config['model'].get('dropout', 0.1)
        
        # 1. Load CLIP Model FIRST to get native dimension
        clip_model_name = config['model']['image_model_name']
        if not clip_model_name:
            raise ValueError("config['model']['image_model_name'] must be specified in config file.")
        logger.info(f"Loading CLIP Model: {clip_model_name}...")
        # Use safetensors to avoid torch.load security vulnerability (CVE-2025-32434).
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        # Cast all weights to float32 so no fp16 residuals remain in the model.
        # autocast(bfloat16) will then downcast cleanly from fp32 during the forward pass,
        # avoiding the at::Half overflow caused by fp16 causal mask fill values (-3.4e38).
        self.clip = self.clip.float()
        
        # Get CLIP's native projection dimension
        self.clip_embed_dim = self.clip.config.projection_dim
        logger.info(f"CLIP Projection Dimension: {self.clip_embed_dim}")
        
        # 2. Determine target embedding dimension
        config_embed_dim = config['model'].get('embed_dim')
        
        # Auto-dimension logic: Use CLIP's native dimension if config is None or matches
        if config_embed_dim is None or config_embed_dim == self.clip_embed_dim:
            self.embed_dim = self.clip_embed_dim
            logger.info(f"Using raw CLIP features (Dim: {self.embed_dim}). No extra projection.")
            self.image_proj = None
            self.text_proj = None
        else:
            # Config explicitly requires a different dimension - create projection layers
            self.embed_dim = config_embed_dim
            logger.warning(
                f"WARNING: Projecting CLIP features from {self.clip_embed_dim} to {self.embed_dim}. "
                "These layers are randomly initialized!"
            )
            self.image_proj = nn.Sequential(
                nn.Linear(self.clip_embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.clip_embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )
            self._init_head_weights()
        
        # 3. Freezing Strategy - Freeze backbone, train projections
        self._apply_freezing_strategy()

        # 4. Reset logit_scale to fine-tuning starting value
        # Pretrained CLIP has logit_scale ~4.6 (τ≈0.01), too high for fine-tuning.
        with torch.no_grad():
            self.clip.logit_scale.fill_(2.6593)  # ln(1/0.07) = 2.6593
        logger.info("logit_scale reset to 2.6593 (τ=0.07)")

        # 5. B5_seg only: segment embedding table for SAM-derived patch IDs.
        # Strictly gated by config.model.seg_embed_size; absent for all other
        # variants so that nothing in the forward path changes for them.
        seg_embed_size = config.get('model', {}).get('seg_embed_size')
        if seg_embed_size is not None:
            if not isinstance(seg_embed_size, int) or seg_embed_size < 1:
                raise ValueError(
                    f"model.seg_embed_size must be a positive int, got {seg_embed_size!r}"
                )
            d_model = self.clip.vision_model.config.hidden_size  # 1024 for ViT-L
            self.seg_embed_size = seg_embed_size
            self.seg_embedding = nn.Embedding(seg_embed_size, d_model)
            # Zero-init: at step 0 the segment embedding contributes nothing,
            # so the model starts identical to B0+ and learns the contribution.
            nn.init.zeros_(self.seg_embedding.weight)
            # Always trainable, even when the rest of the vision encoder is
            # frozen. This is a new parameter, not part of CLIP.
            self.seg_embedding.weight.requires_grad = True
            # Cache the patch grid layout once for shape assertions.
            patch_size = self.clip.vision_model.config.patch_size
            image_size = self.clip.vision_model.config.image_size
            self._seg_grid_size = image_size // patch_size
            self._seg_num_patches = self._seg_grid_size * self._seg_grid_size
            logger.info(
                f"B5_seg: seg_embedding initialized with n_seg={seg_embed_size}, "
                f"d_model={d_model}, num_patches={self._seg_num_patches} "
                f"(grid {self._seg_grid_size}x{self._seg_grid_size})"
            )
            # Activation memory: seg_embedding sits BEFORE the 24-layer ViT
            # encoder, so even though the encoder is frozen, autograd must
            # retain every layer's activations to backprop into seg_embedding.
            # Enable gradient checkpointing on the vision encoder to drop them.
            # Cost is ~one extra forward per step, which is grad-free here
            # because the encoder weights themselves are frozen.
            # gradient_checkpointing_enable lives on PreTrainedModel (self.clip),
            # not on the inner CLIPVisionTransformer. Calling it on self.clip
            # propagates the flag to vision + text encoders; text is frozen too
            # so the extra recompute there is also grad-free.
            self.clip.gradient_checkpointing_enable()
            self.clip.config.use_cache = False
            logger.info("B5_seg: enabled gradient checkpointing on CLIPModel")
        else:
            self.seg_embed_size = None
            self.seg_embedding = None
    
    def _apply_freezing_strategy(self):
        """
        Freeze CLIP backbone, selectively unfreeze layers for fine-tuning.
        
        CLIP ViT-L/14 architecture:
        - vision_model.encoder.layers.0-23  (24 transformer blocks)
        - text_model.encoder.layers.0-23    (24 transformer blocks)
        - visual_projection: Linear layer
        - text_projection: Linear layer
        
        Strategy: Unfreeze projections + last N vision transformer blocks
        """
        # First, freeze everything
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 1. Unfreeze CLIP's projection layers (always)
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True
        
        # 2. Unfreeze last N transformer blocks of Vision Encoder
        # ViT-L/14 has 24 blocks (0-23)
        # Config: unfreeze_vision_layers = 2 means unfreeze blocks [22, 23]
        num_vision_layers = self.config.get('model', {}).get('unfreeze_vision_layers', 0)
        unfreeze_strategy = self.config.get('model', {}).get('unfreeze_strategy', 'full')
        
        if num_vision_layers > 0:
            # Calculate which blocks to unfreeze (from the end)
            # Dynamically get total blocks from the model
            total_blocks = len(self.clip.vision_model.encoder.layers)
            unfreeze_vision_blocks = list(range(total_blocks - num_vision_layers, total_blocks))
            logger.info(f"Unfreezing Vision Blocks: {unfreeze_vision_blocks} (strategy: {unfreeze_strategy})")
            
            # Directly access and unfreeze the specified blocks
            for block_idx in unfreeze_vision_blocks:
                block = self.clip.vision_model.encoder.layers[block_idx]
                for name, param in block.named_parameters():
                    # Apply partial unfreezing strategy
                    should_unfreeze = self._should_unfreeze_param(name, unfreeze_strategy)
                    if should_unfreeze:
                        param.requires_grad = True
        
        # 3. Unfreeze vision model's post_layernorm (only if vision blocks are unfrozen)
        if num_vision_layers > 0 and unfreeze_strategy in ['full', 'layernorm']:
            for param in self.clip.vision_model.post_layernorm.parameters():
                param.requires_grad = True
        
        # 4. Unfreeze CLIP's learnable temperature (logit_scale)
        # This is critical for proper contrastive learning!
        if hasattr(self.clip, 'logit_scale'):
            self.clip.logit_scale.requires_grad = True
        
        # Print summary
        self._print_freezing_summary()
    
    def _should_unfreeze_param(self, name: str, strategy: str) -> bool:
        """
        Determine if a parameter should be unfrozen based on strategy.
        
        Strategies:
        - "full": Unfreeze entire block (all parameters)
        - "attention": Only Q, K, V, and output projections in self-attention
        - "mlp": Only MLP/FFN layers (fc1, fc2)
        - "layernorm": Only LayerNorm parameters (very lightweight)
        - "bias": Only bias terms (BitFit style, extremely lightweight)
        
        ViT layer naming convention:
        - encoder.layers.X.self_attn.q_proj, k_proj, v_proj, out_proj
        - encoder.layers.X.mlp.fc1, fc2
        - encoder.layers.X.layer_norm1, layer_norm2
        """
        if strategy == "full":
            return True
        
        elif strategy == "attention":
            # Unfreeze: q_proj, k_proj, v_proj, out_proj
            attention_keywords = ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
            return any(kw in name for kw in attention_keywords)
        
        elif strategy == "mlp":
            # Unfreeze: fc1, fc2 (MLP layers)
            mlp_keywords = ['mlp', 'fc1', 'fc2']
            return any(kw in name for kw in mlp_keywords)
        
        elif strategy == "layernorm":
            # Unfreeze: layer_norm1, layer_norm2, LayerNorm
            return 'layer_norm' in name.lower() or 'layernorm' in name.lower()
        
        elif strategy == "bias":
            # Unfreeze: Only bias parameters (BitFit)
            return 'bias' in name
        
        else:
            logger.warning(f"Unknown unfreeze strategy '{strategy}', defaulting to 'full'")
            return True
    
    def _print_freezing_summary(self):
        """Print trainable parameter summary."""
        total_params = sum(p.numel() for p in self.clip.parameters())
        trainable_params = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        
        logger.info(f"CLIP: {trainable_params:,} trainable / {total_params:,} total params")
        logger.info("Frozen: Vision Encoder (partial), Text Encoder | Unfrozen: Projections + Selected")
    
    def _init_head_weights(self):
        """Initialize additional projection heads."""
        for proj in [self.image_proj, self.text_proj]:
            if proj is not None:
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def _get_image_features(self, images, seg_ids=None):
        """
        Extract image features and standardize embeddings to float32.

        Behavior matrix:
            seg_embedding | seg_ids | path
            --------------+---------+--------------------------------------
            None          | None    | standard CLIP (B0/B0+/B1/B2/B3/FULL)
            None          | tensor  | ValueError (gating mismatch)
            set (B5_seg)  | None    | standard CLIP (no seg_emb injection)
            set (B5_seg)  | tensor  | explicit forward with seg_emb injection

        The (set, None) cell is intentional: the augmented intra-modal view
        uses RandomResizedCrop(scale=(0.4, 1.0)) which destroys spatial
        alignment with the precomputed segment map. Passing seg_ids to that
        view would inject misleading "this patch is dog" signal onto a sky
        crop. The anchor view (mild crop) keeps the seg_ids; the augmented
        view drops them and runs through standard CLIP, identical to B0+.
        """
        if seg_ids is None:
            # Standard CLIP path. Used for every non-B5_seg variant AND for
            # the augmented view of B5_seg (see docstring).
            image_embeds = self.clip.get_image_features(pixel_values=images)
            return image_embeds.float()

        if self.seg_embedding is None:
            raise ValueError(
                "seg_ids was provided but model.seg_embed_size is not set. "
                "Either configure B5_seg or stop passing seg_ids."
            )

        # --- B5_seg path: explicit re-implementation of CLIPVisionTransformer
        # forward (transformers 4.48.0) with segment-embedding injection. ---
        if seg_ids.dtype != torch.long:
            raise TypeError(
                f"seg_ids must be torch.long, got {seg_ids.dtype}. "
                "Cast at the dataloader, not here."
            )
        if seg_ids.dim() != 2 or seg_ids.size(0) != images.size(0) \
                or seg_ids.size(1) != self._seg_num_patches:
            raise ValueError(
                f"seg_ids has shape {tuple(seg_ids.shape)}, expected "
                f"({images.size(0)}, {self._seg_num_patches})."
            )
        if seg_ids.min().item() < 0 or seg_ids.max().item() >= self.seg_embed_size:
            raise ValueError(
                f"seg_ids out of range [0, {self.seg_embed_size}): "
                f"min={seg_ids.min().item()}, max={seg_ids.max().item()}"
            )

        vm = self.clip.vision_model
        # Stage 1: patch + position embeddings.
        hidden_states = vm.embeddings(images)  # [B, 1+576, 1024]

        # Stage 2: build full sequence of segment IDs with CLS=0 (background).
        # The CLS token (index 0) is treated as background per the spec.
        cls_seg = torch.zeros(
            seg_ids.size(0), 1, dtype=torch.long, device=seg_ids.device
        )
        full_seg_ids = torch.cat([cls_seg, seg_ids], dim=1)  # [B, 577]
        if full_seg_ids.size(1) != hidden_states.size(1):
            raise RuntimeError(
                f"Sequence length mismatch: hidden_states={hidden_states.size(1)}, "
                f"full_seg_ids={full_seg_ids.size(1)}. CLIP vision_model.embeddings "
                "produced an unexpected number of tokens."
            )

        # Stage 3: inject segment embeddings (additive). seg_embedding is
        # zero-initialized, so this is a no-op at step 0.
        seg_emb = self.seg_embedding(full_seg_ids)  # [B, 577, 1024]
        hidden_states = hidden_states + seg_emb

        # Stage 4: continue through pre_layrnorm -> encoder -> CLS pool ->
        # post_layernorm -> visual_projection. Mirrors transformers 4.48.0
        # CLIPVisionTransformer.forward exactly. The attribute name is
        # "pre_layrnorm" (typo preserved upstream for backward compat).
        hidden_states = vm.pre_layrnorm(hidden_states)
        encoder_outputs = vm.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]            # [B, 1024]   CLS token
        pooled_output = vm.post_layernorm(pooled_output)      # [B, 1024]
        image_features = self.clip.visual_projection(pooled_output)  # [B, embed_dim]
        return image_features.float()

    def _get_text_features(self, input_ids, attention_mask):
        """Extract text features and standardize embeddings to float32."""
        text_embeds = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return text_embeds.float()

    def forward(self, images, input_ids, attention_mask, seg_ids=None):
        """
        Forward pass returning L2-normalized embeddings.

        Args:
            images: [Batch, 3, 336, 336] - Pixel values (336px for CLIP ViT-L/14-336)
            input_ids: [Batch, SeqLen] - Tokenized text
            attention_mask: [Batch, SeqLen] - Attention mask
            seg_ids: [Batch, 576] LongTensor - Per-patch SAM segment IDs.
                     REQUIRED iff config.model.seg_embed_size is set, must be
                     None otherwise (B5_seg variant only).

        Returns:
            img_embeds: [Batch, embed_dim] - L2 normalized image embeddings
            txt_embeds: [Batch, embed_dim] - L2 normalized text embeddings
        """
        # Get CLIP embeddings (already projected)
        # Note: CLIP expects 'pixel_values' for images
        image_embeds = self._get_image_features(images, seg_ids=seg_ids)
        text_embeds = self._get_text_features(input_ids, attention_mask)

        # Optional: Additional projection to match target embed_dim
        if self.image_proj is not None:
            image_embeds = self.image_proj(image_embeds)
            text_embeds = self.text_proj(text_embeds)

        # L2 Normalization (critical for contrastive learning)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        return image_embeds, text_embeds

    def encode_image(self, images, seg_ids=None):
        """
        Encode images only (L2-normalized).

        Args:
            images: [B, 3, 336, 336]
            seg_ids: [B, 576] LongTensor - REQUIRED iff config.model.seg_embed_size
                     is set, must be None otherwise (B5_seg variant only).
        """
        image_embeds = self._get_image_features(images, seg_ids=seg_ids)
        if self.image_proj is not None:
            image_embeds = self.image_proj(image_embeds)
        return F.normalize(image_embeds, p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        """Encode text only (L2-normalized). For use with separate image encoding (e.g. hard negatives)."""
        text_embeds = self._get_text_features(input_ids, attention_mask)
        if self.text_proj is not None:
            text_embeds = self.text_proj(text_embeds)
        return F.normalize(text_embeds, p=2, dim=1)

