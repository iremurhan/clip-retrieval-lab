"""
src/grad_cache.py
-----------------
Gradient Caching for Contrastive Learning (CLIP-style).

Mechanism:
    Gradient Caching decouples batch size from GPU memory by splitting the forward
    and backward passes:
    
    1. **Cache Phase**: Compute embeddings for ALL samples in the batch using 
       micro-batches, WITHOUT building computation graphs (no_grad). Store embeddings.
    
    2. **Gradient Phase**: For each micro-batch:
       - Re-compute embeddings WITH gradients for the current micro-batch only.
       - Use cached (detached) embeddings for all other samples.
       - Compute loss on the full similarity matrix.
       - Backward pass updates gradients ONLY for the current micro-batch.
    
    This allows simulating large batch contrastive learning (e.g., 512) on limited
    VRAM (e.g., micro_batch=32, requiring only 32-sample memory for gradients).

Mathematical Correctness:
    The gradient of the contrastive loss w.r.t. a sample depends on:
    - The sample's own embedding
    - ALL other embeddings in the batch (for negative pairs)
    
    By caching embeddings and recomputing only the current micro-batch with gradients,
    we compute the exact same gradient as if we processed the entire batch at once.

Reference: Gao et al., "Scaling Deep Contrastive Learning..." (GradCache paper)
           Implementation inspired by luyug/GradCache but customized for strict config.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Caching for CLIP-style contrastive training.
    
    Supports:
    - Mixed Precision (AMP) via GradScaler
    - Strict configuration (raises KeyError on missing params)
    - Dictionary-based loss returns
    
    Args:
        model: The CLIP model with encode_image() and encode_text() methods
        criterion: Loss function (SymmetricInfoNCELoss)
        config: Configuration dict (must contain training.micro_batch_size)
        device: torch.device
        scaler: torch.cuda.amp.GradScaler or None (for AMP)
    """
    
    def __init__(self, model, criterion, config, device, scaler=None):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        self.scaler = scaler
        
        # STRICT CONFIG: No fallback, must crash if missing
        self.micro_batch_size = config['training']['micro_batch_size']
        
        # Check if AMP is enabled
        self.use_amp = scaler is not None
        
        logger.info(f"GradCache initialized: micro_batch_size={self.micro_batch_size}, AMP={self.use_amp}")
    
    def _split_into_chunks(self, tensor, chunk_size):
        """Split a tensor into chunks of size chunk_size."""
        return [tensor[i:i + chunk_size] for i in range(0, tensor.size(0), chunk_size)]
    
    def forward(self, images, input_ids, attention_mask):
        """
        Perform gradient-cached forward and backward pass.
        
        Args:
            images: [N, C, H, W] - Full batch of images
            input_ids: [N, L] - Full batch of text input IDs
            attention_mask: [N, L] - Full batch of attention masks
        
        Returns:
            dict: Loss dictionary with 'loss_total' and component losses
        """
        batch_size = images.size(0)
        
        if batch_size <= self.micro_batch_size:
            # No need for gradient caching if batch fits in micro_batch
            logger.debug("Batch size <= micro_batch_size, using standard forward pass")
            return self._standard_forward(images, input_ids, attention_mask)
        
        # ============================================================
        # PHASE 1: CACHE - Compute all embeddings without gradients
        # ============================================================
        img_chunks = self._split_into_chunks(images, self.micro_batch_size)
        txt_input_chunks = self._split_into_chunks(input_ids, self.micro_batch_size)
        txt_mask_chunks = self._split_into_chunks(attention_mask, self.micro_batch_size)
        
        cached_img_embeds = []
        cached_txt_embeds = []
        
        with torch.no_grad():
            for img_chunk, txt_input_chunk, txt_mask_chunk in zip(img_chunks, txt_input_chunks, txt_mask_chunks):
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        img_emb = self.model.encode_image(img_chunk)
                        txt_emb = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
                else:
                    img_emb = self.model.encode_image(img_chunk)
                    txt_emb = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
                
                cached_img_embeds.append(img_emb.detach())
                cached_txt_embeds.append(txt_emb.detach())
        
        # Concatenate all cached embeddings
        cached_img_embeds = torch.cat(cached_img_embeds, dim=0)
        cached_txt_embeds = torch.cat(cached_txt_embeds, dim=0)
        
        # ============================================================
        # PHASE 2: GRADIENT - Recompute each micro-batch with gradients
        # ============================================================
        total_loss = 0.0
        loss_dict_accumulator = {
            "loss_total": 0.0,
            "loss_inter": 0.0,
            "loss_intra_img": 0.0,
            "loss_intra_txt": 0.0
        }
        
        num_chunks = len(img_chunks)
        
        for chunk_idx, (img_chunk, txt_input_chunk, txt_mask_chunk) in enumerate(
            zip(img_chunks, txt_input_chunks, txt_mask_chunks)
        ):
            # Re-compute embeddings WITH gradients for current micro-batch
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    img_emb_grad = self.model.encode_image(img_chunk)
                    txt_emb_grad = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
            else:
                img_emb_grad = self.model.encode_image(img_chunk)
                txt_emb_grad = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
            
            # Build full-batch embeddings: current chunk has gradients, others are cached (detached)
            chunk_start = chunk_idx * self.micro_batch_size
            chunk_end = chunk_start + img_chunk.size(0)

            img_embeds_full = torch.cat([cached_img_embeds[:chunk_start], img_emb_grad, cached_img_embeds[chunk_end:]], dim=0)
            txt_embeds_full = torch.cat([cached_txt_embeds[:chunk_start], txt_emb_grad, cached_txt_embeds[chunk_end:]], dim=0)
            
            # Compute loss on full batch (but only current chunk has gradients)
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    loss_dict = self.criterion(
                        img_embeds_full, 
                        txt_embeds_full,
                        img_aug_embeds=None,  # Intra-modal not supported yet in grad cache
                        txt_aug_embeds=None
                    )
            else:
                loss_dict = self.criterion(
                    img_embeds_full, 
                    txt_embeds_full,
                    img_aug_embeds=None,
                    txt_aug_embeds=None
                )
            
            # NOTE: We do NOT scale the loss here. The full batch loss is computed,
            # but only the gradients for the current micro-batch will flow backward.
            # This is mathematically correct for gradient caching.
            loss = loss_dict["loss_total"]
            
            # Backward pass (only current chunk gradients are updated)
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate loss components for logging (averaged across chunks)
            for key in loss_dict_accumulator:
                loss_dict_accumulator[key] += loss_dict[key].item() / num_chunks
        
        # Return averaged loss dictionary (for logging)
        # Convert back to tensors for consistency with standard forward
        return {
            key: torch.tensor(value, device=self.device) 
            for key, value in loss_dict_accumulator.items()
        }
    
    def _standard_forward(self, images, input_ids, attention_mask):
        """Standard forward pass when gradient caching is not needed."""
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda'):
                img_embeds = self.model.encode_image(images)
                txt_embeds = self.model.encode_text(input_ids, attention_mask)
                loss_dict = self.criterion(
                    img_embeds, txt_embeds,
                    img_aug_embeds=None,
                    txt_aug_embeds=None
                )
        else:
            img_embeds = self.model.encode_image(images)
            txt_embeds = self.model.encode_text(input_ids, attention_mask)
            loss_dict = self.criterion(
                img_embeds, txt_embeds,
                img_aug_embeds=None,
                txt_aug_embeds=None
            )
        
        return loss_dict
