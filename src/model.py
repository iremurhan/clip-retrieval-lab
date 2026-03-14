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


class CrossAttentionFusion(nn.Module):
    """
    Text-Guided Cross-Attention Fusion for Vision-Language models.

    This module fuses image patch embeddings with a text query using cross-attention.
    The text pooled representation acts as the query (Q), while image ViT patches serve as both
    keys (K) and values (V).

    Architecture:
    - Input: Text embeddings (Q) [B, 1, text_hidden_dim] and image patch embeddings (K, V) [B, N+1, vision_hidden_dim]
    - Project text query to vision_hidden_dim (intermediate space for alignment)
    - Project image patches to vision_hidden_dim
    - Compute attention: Attention(Q,K,V) = softmax((Q*K^T) / sqrt(d)) * V
    - Output: Fused image representation [B, vision_hidden_dim] and attention map [B, 1, N]

    The attention mechanism learns which image patches are most relevant to the text query,
    enabling fine-grained cross-modal alignment.
    """

    def __init__(self, vision_hidden_dim, text_hidden_dim):
        """
        Initialize CrossAttentionFusion with dynamic dimension handling.

        Args:
            vision_hidden_dim (int): Hidden dimension from ViT encoder (e.g., 1024 for ViT-L/14)
            text_hidden_dim (int): Hidden dimension from text encoder (e.g., 1024 for CLIP text model)
        """
        super().__init__()
        self.vision_hidden_dim = vision_hidden_dim
        self.text_hidden_dim = text_hidden_dim

        # Linear projections to align text and vision dimensions
        # Project text query from text_hidden_dim to vision_hidden_dim for cross-modal matching
        self.q_proj = nn.Linear(text_hidden_dim, vision_hidden_dim)
        # Project image patches within vision_hidden_dim space
        self.k_proj = nn.Linear(vision_hidden_dim, vision_hidden_dim)
        self.v_proj = nn.Linear(vision_hidden_dim, vision_hidden_dim)

        # Output projection to refine fused representation
        self.out_proj = nn.Linear(vision_hidden_dim, vision_hidden_dim)

        # Scaling factor for attention (inverse sqrt of dimension for numerical stability)
        self.scale = (vision_hidden_dim ** -0.5)

    def forward(self, text_emb, image_patches):
        """
        Compute cross-attention fusion.

        Args:
            text_emb: [B, 1, text_hidden_dim] - Unnormalized text representation from pooler
            image_patches: [B, N+1, vision_hidden_dim] - Image ViT patch embeddings including [CLS] token

        Returns:
            fused_image: [B, vision_hidden_dim] - Fused image representation (squeezed)
            attn_probs: [B, 1, N] - Attention probability matrix for patches only (excluding [CLS])
        """
        batch_size, num_tokens, vision_dim = image_patches.shape
        num_patches = num_tokens - 1  # Exclude [CLS] token (assumed to be at index 0)

        # Project Q, K, V to shared vision dimension
        Q = self.q_proj(text_emb)  # [B, 1, vision_hidden_dim]
        K = self.k_proj(image_patches)  # [B, N+1, vision_hidden_dim]
        V = self.v_proj(image_patches)  # [B, N+1, vision_hidden_dim]

        # Compute attention scores: (Q @ K^T) / sqrt(d)
        # Q: [B, 1, D] x K.T: [B, D, N+1] -> [B, 1, N+1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, 1, N+1]

        # Apply softmax to get attention weights over all tokens (including [CLS])
        # This ensures maximum information flow from all image regions to the text query
        attn_probs_full = F.softmax(scores, dim=-1)  # [B, 1, N+1]

        # Apply attention to values: attn_probs_full @ V
        # [B, 1, N+1] x [B, N+1, D] -> [B, 1, D]
        fused = torch.matmul(attn_probs_full, V)  # [B, 1, vision_hidden_dim]

        # Apply output projection for additional representational capacity
        fused = self.out_proj(fused)  # [B, 1, vision_hidden_dim]

        # Squeeze to [B, D] for downstream processing
        fused_squeezed = fused.squeeze(1)  # [B, vision_hidden_dim]

        # Extract attention probabilities for patches only (exclude [CLS] at index 0)
        # This provides clean visualization without [CLS] token for W&B logging
        # Note: For 336px input with 24x24 patch grid, reshape [B, 1, 576] to [B, 24, 24] during visualization
        attn_probs = attn_probs_full[:, :, 1:]  # [B, 1, N]

        return fused_squeezed, attn_probs


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
        # Use safetensors to avoid torch.load security vulnerability (CVE-2025-32434)
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)

        # Retrieve hidden dimensions from both encoders for dynamic architecture handling
        # Vision encoder hidden dimension (e.g., 1024 for ViT-L/14-336)
        self.vision_hidden_dim = self.clip.vision_model.config.hidden_size
        logger.info(f"Vision Hidden Dimension: {self.vision_hidden_dim}")
        
        # Text encoder hidden dimension (e.g., 1024 for CLIP text tower)
        self.text_hidden_dim = self.clip.text_model.config.hidden_size
        logger.info(f"Text Hidden Dimension: {self.text_hidden_dim}")
        
        # Get CLIP's projection dimension for optional downstream projection heads
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

        # 3. Initialize Cross-Attention Fusion for text-guided image pooling
        # Pass both vision and text hidden dimensions for dynamic alignment
        self.cross_attn_fusion = CrossAttentionFusion(
            vision_hidden_dim=self.vision_hidden_dim,
            text_hidden_dim=self.text_hidden_dim
        )
        logger.info(f"CrossAttentionFusion initialized: vision_dim={self.vision_hidden_dim}, text_dim={self.text_hidden_dim}")

        # 4. Freezing Strategy - Freeze backbone, train projections
        self._apply_freezing_strategy()
    
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
            print(f"  Unfreezing Vision Blocks: {unfreeze_vision_blocks} (strategy: {unfreeze_strategy})")
            
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
            print(f"  Warning: Unknown unfreeze strategy '{strategy}', defaulting to 'full'")
            return True
    
    def _print_freezing_summary(self):
        """Print trainable parameter summary."""
        total_params = sum(p.numel() for p in self.clip.parameters())
        trainable_params = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        
        print(f"  CLIP: {trainable_params:,} trainable / {total_params:,} total params")
        print(f"  Frozen: Vision Encoder (partial), Text Encoder | Unfrozen: Projections + Selected")
    
    def _init_head_weights(self):
        """Initialize additional projection heads."""
        for proj in [self.image_proj, self.text_proj]:
            if proj is not None:
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass returning L2-normalized embeddings with text-guided image fusion.

        Args:
            images: [Batch, 3, 336, 336] - Pixel values (336px for CLIP ViT-L/14-336)
            input_ids: [Batch, SeqLen] - Tokenized text
            attention_mask: [Batch, SeqLen] - Attention mask

        Returns:
            img_embeds: [Batch, embed_dim] - L2 normalized image embeddings (text-guided)
            txt_embeds: [Batch, embed_dim] - L2 normalized text embeddings
        """
        # Image encoding with text-guided cross-attention fusion
        img_embeds, _ = self.encode_image(images, input_ids, attention_mask)

        # Text encoding
        txt_embeds = self.encode_text(input_ids, attention_mask)

        return img_embeds, txt_embeds

    def encode_image(self, images, input_ids=None, attention_mask=None):
        """
        Encode images with optional text-guided cross-attention fusion.

        When text inputs are provided, the image is pooled using text-guided attention,
        enabling fine-grained cross-modal alignment. Otherwise, standard CLIP pooling is used.

        Args:
            images: [B, 3, H, W] - Pixel values
            input_ids: [B, L] optional - Text input IDs for text-guided fusion
            attention_mask: [B, L] optional - Text attention mask

        Returns:
            image_embeds: [B, embed_dim] - L2 normalized image embeddings
            attn_probs: [B, 1, N] optional - Attention probabilities if text provided, else None
        """
        # Get raw image features from CLIP vision model
        # Note: This returns the full vision_model outputs which include patch embeddings
        vision_outputs = self.clip.vision_model(pixel_values=images)
        # vision_outputs.last_hidden_state shape: [B, N+1, 768] where N is number of patches and +1 is [CLS]
        image_patches_with_cls = vision_outputs.last_hidden_state  # [B, N+1, 768]

        # Text-guided fusion case
        if input_ids is not None and attention_mask is not None:
            # Get unnormalized text representation via EOS token from last_hidden_state
            # CLIP text model has no pooler head; EOS token (highest token id) is the pooled rep
            text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = text_outputs.last_hidden_state  # [B, L, text_hidden_dim]
            eos_pos = input_ids.argmax(dim=-1)  # [B]
            text_pooler_output = last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), eos_pos]  # [B, text_hidden_dim]
            text_embeds = text_pooler_output.unsqueeze(1)  # [B, 1, text_hidden_dim]

            # Apply cross-attention fusion
            # Use all image patches (including [CLS]) as K, V for maximum information flow
            # Returns: pooled embedding [B, vision_hidden_dim] and attention probs for patches only [B, 1, N]
            # Attention computed over all tokens for full context, but [CLS] excluded from returned probs for visualization
            fused_image_embeds, attn_probs = self.cross_attn_fusion(text_embeds, image_patches_with_cls)  # [B, vision_hidden_dim], [B, 1, N]

            # Project from vision_hidden_dim to clip_embed_dim if difference exists
            # Then apply optional projection to target embed_dim
            if self.vision_hidden_dim != self.clip_embed_dim:
                # Use CLIP's visual projection layer to go from vision_hidden_dim to clip_embed_dim
                fused_image_embeds = self.clip.visual_projection(fused_image_embeds)

            if self.image_proj is not None:
                fused_image_embeds = self.image_proj(fused_image_embeds)

            # Normalize and return with attention probs for W&B visualization
            # Note: attn_probs shape is [B, 1, N] where N=576 for 336px input
            # Reshape to [B, 24, 24] during visualization for 2D heatmap overlay
            return F.normalize(fused_image_embeds, p=2, dim=1), attn_probs

        # Standard CLIP pooling case (no text guidance)
        else:
            # Use CLIP's standard approach: linear projection of [CLS] token
            image_embeds = self.clip.visual_projection(image_patches_with_cls[:, 0, :])  # [B, 768]

            # Apply optional projection
            if self.image_proj is not None:
                image_embeds = self.image_proj(image_embeds)

            # Normalize and return without attention probs
            return F.normalize(image_embeds, p=2, dim=1), None

    def encode_text(self, input_ids, attention_mask):
        """Encode text only (L2-normalized). For use with separate image encoding (e.g. hard negatives)."""
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # CLIP uses EOS token (highest input_id position) as the pooled representation
        last_hidden = text_outputs.last_hidden_state  # [B, L, text_hidden_dim]
        eos_pos = input_ids.argmax(dim=-1)  # [B] — EOS token has the highest token id
        text_embeds = last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), eos_pos]  # [B, text_hidden_dim]
        text_embeds = self.clip.text_projection(text_embeds)  # [B, clip_embed_dim]
        if self.text_proj is not None:
            text_embeds = self.text_proj(text_embeds)
        return F.normalize(text_embeds, p=2, dim=1)

    def forward_with_clip_loss(self, images, input_ids, attention_mask):
        """
        Alternative forward that returns CLIP's built-in contrastive loss.
        
        Use this if you want to bypass custom loss function.
        Note: This ignores our additional projection heads.
        
        Returns:
            loss: Scalar contrastive loss computed by CLIP
            logits_per_image: [Batch, Batch] similarity matrix
            logits_per_text: [Batch, Batch] similarity matrix (transposed)
        """
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
            return_loss=True
        )
        return outputs.loss, outputs.logits_per_image, outputs.logits_per_text
