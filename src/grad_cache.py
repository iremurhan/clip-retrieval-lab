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
        self.amp_dtype = torch.bfloat16 if (scaler is not None and self.device.type == 'cuda') else None
        
        # STRICT CONFIG: No fallback, must crash if missing
        self.micro_batch_size = config['training']['micro_batch_size']
        
        # Check if AMP is enabled
        self.use_amp = scaler is not None
        
        logger.info(f"GradCache initialized: micro_batch_size={self.micro_batch_size}, AMP={self.use_amp}")
    
    def _split_into_chunks(self, tensor, chunk_size):
        """Split a tensor into chunks of size chunk_size."""
        return [tensor[i:i + chunk_size] for i in range(0, tensor.size(0), chunk_size)]
    
    def forward(self, images, input_ids, attention_mask,
                para_input_ids_a=None, para_attention_mask_a=None,
                para_input_ids_b=None, para_attention_mask_b=None,
                image_aug_a=None, image_aug_b=None):
        """
        Perform gradient-cached forward and backward pass.

        Args:
            images: [N, C, H, W] - Full batch of images
            input_ids: [N, L] - Full batch of text input IDs
            attention_mask: [N, L] - Full batch of attention masks
            para_input_ids_a/b: [N, 77] optional - paired paraphrase token IDs
                for L_text_text.
            para_attention_mask_a/b: [N, 77] optional - paired paraphrase masks.
            image_aug_a/b: [N, C, H, W] optional - paired augmented images for
                L_img_img.

        Returns:
            dict: Loss dictionary with 'loss_total' and component losses
        """
        batch_size = images.size(0)

        if batch_size <= self.micro_batch_size:
            # No need for gradient caching if batch fits in micro_batch
            logger.debug("Batch size <= micro_batch_size, using standard forward pass")
            return self._standard_forward(
                images, input_ids, attention_mask,
                para_input_ids_a=para_input_ids_a,
                para_attention_mask_a=para_attention_mask_a,
                para_input_ids_b=para_input_ids_b,
                para_attention_mask_b=para_attention_mask_b,
                image_aug_a=image_aug_a,
                image_aug_b=image_aug_b,
            )

        # ============================================================
        # PHASE 1: CACHE - Compute all embeddings without gradients
        # ============================================================
        img_chunks = self._split_into_chunks(images, self.micro_batch_size)
        txt_input_chunks = self._split_into_chunks(input_ids, self.micro_batch_size)
        txt_mask_chunks = self._split_into_chunks(attention_mask, self.micro_batch_size)

        # Paired paraphrase chunks: split if provided, else fill with None placeholders.
        w_txt = getattr(self.criterion, 'w_txt', 0)
        has_para_pair = (
            para_input_ids_a is not None
            and para_attention_mask_a is not None
            and para_input_ids_b is not None
            and para_attention_mask_b is not None
            and w_txt > 0
        )
        para_input_a_chunks = (
            self._split_into_chunks(para_input_ids_a, self.micro_batch_size)
            if has_para_pair else [None] * len(img_chunks)
        )
        para_mask_a_chunks = (
            self._split_into_chunks(para_attention_mask_a, self.micro_batch_size)
            if has_para_pair else [None] * len(img_chunks)
        )
        para_input_b_chunks = (
            self._split_into_chunks(para_input_ids_b, self.micro_batch_size)
            if has_para_pair else [None] * len(img_chunks)
        )
        para_mask_b_chunks = (
            self._split_into_chunks(para_attention_mask_b, self.micro_batch_size)
            if has_para_pair else [None] * len(img_chunks)
        )

        # Paired image augmentation chunks: split if provided, else fill with placeholders.
        w_img = getattr(self.criterion, 'w_img', 0)
        has_img_aug_pair = (image_aug_a is not None) and (image_aug_b is not None) and (w_img > 0)
        img_aug_a_chunks = (
            self._split_into_chunks(image_aug_a, self.micro_batch_size)
            if has_img_aug_pair else [None] * len(img_chunks)
        )
        img_aug_b_chunks = (
            self._split_into_chunks(image_aug_b, self.micro_batch_size)
            if has_img_aug_pair else [None] * len(img_chunks)
        )

        cached_img_embeds = []
        cached_txt_embeds = []
        cached_para_a_embeds = []
        cached_para_b_embeds = []
        cached_img_aug_a_embeds = []
        cached_img_aug_b_embeds = []

        with torch.no_grad():
            for (
                img_chunk,
                txt_input_chunk,
                txt_mask_chunk,
                para_in_a_chunk,
                para_mask_a_chunk,
                para_in_b_chunk,
                para_mask_b_chunk,
                img_aug_a_chunk,
                img_aug_b_chunk,
            ) in zip(
                img_chunks,
                txt_input_chunks,
                txt_mask_chunks,
                para_input_a_chunks,
                para_mask_a_chunks,
                para_input_b_chunks,
                para_mask_b_chunks,
                img_aug_a_chunks,
                img_aug_b_chunks,
            ):
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        img_emb = self.model.encode_image(img_chunk)
                        txt_emb = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
                        if has_para_pair:
                            para_a_emb = self.model.encode_text(para_in_a_chunk, para_mask_a_chunk)
                            para_b_emb = self.model.encode_text(para_in_b_chunk, para_mask_b_chunk)
                        if has_img_aug_pair:
                            img_aug_a_emb = self.model.encode_image(img_aug_a_chunk)
                            img_aug_b_emb = self.model.encode_image(img_aug_b_chunk)
                else:
                    img_emb = self.model.encode_image(img_chunk)
                    txt_emb = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
                    if has_para_pair:
                        para_a_emb = self.model.encode_text(para_in_a_chunk, para_mask_a_chunk)
                        para_b_emb = self.model.encode_text(para_in_b_chunk, para_mask_b_chunk)
                    if has_img_aug_pair:
                        img_aug_a_emb = self.model.encode_image(img_aug_a_chunk)
                        img_aug_b_emb = self.model.encode_image(img_aug_b_chunk)

                cached_img_embeds.append(img_emb.detach())
                cached_txt_embeds.append(txt_emb.detach())
                if has_para_pair:
                    cached_para_a_embeds.append(para_a_emb.detach())
                    cached_para_b_embeds.append(para_b_emb.detach())
                if has_img_aug_pair:
                    cached_img_aug_a_embeds.append(img_aug_a_emb.detach())
                    cached_img_aug_b_embeds.append(img_aug_b_emb.detach())

        # Concatenate all cached embeddings
        cached_img_embeds = torch.cat(cached_img_embeds, dim=0)      # [N, D]
        cached_txt_embeds = torch.cat(cached_txt_embeds, dim=0)      # [N, D]
        cached_para_a_embeds_full = (
            torch.cat(cached_para_a_embeds, dim=0) if has_para_pair else None
        )  # [N, D] or None
        cached_para_b_embeds_full = (
            torch.cat(cached_para_b_embeds, dim=0) if has_para_pair else None
        )  # [N, D] or None
        cached_img_aug_a_embeds_full = (
            torch.cat(cached_img_aug_a_embeds, dim=0) if has_img_aug_pair else None
        )  # [N, D] or None
        cached_img_aug_b_embeds_full = (
            torch.cat(cached_img_aug_b_embeds, dim=0) if has_img_aug_pair else None
        )  # [N, D] or None

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

        for chunk_idx, (
            img_chunk,
            txt_input_chunk,
            txt_mask_chunk,
            para_in_a_chunk,
            para_mask_a_chunk,
            para_in_b_chunk,
            para_mask_b_chunk,
            img_aug_a_chunk,
            img_aug_b_chunk,
        ) in enumerate(
            zip(
                img_chunks,
                txt_input_chunks,
                txt_mask_chunks,
                para_input_a_chunks,
                para_mask_a_chunks,
                para_input_b_chunks,
                para_mask_b_chunks,
                img_aug_a_chunks,
                img_aug_b_chunks,
            )
        ):
            # Re-compute embeddings WITH gradients for current micro-batch
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    img_emb_grad = self.model.encode_image(img_chunk)
                    txt_emb_grad = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
                    if has_para_pair:
                        para_a_emb_grad = self.model.encode_text(para_in_a_chunk, para_mask_a_chunk)
                        para_b_emb_grad = self.model.encode_text(para_in_b_chunk, para_mask_b_chunk)
                    if has_img_aug_pair:
                        img_aug_a_emb_grad = self.model.encode_image(img_aug_a_chunk)
                        img_aug_b_emb_grad = self.model.encode_image(img_aug_b_chunk)
            else:
                img_emb_grad = self.model.encode_image(img_chunk)
                txt_emb_grad = self.model.encode_text(txt_input_chunk, txt_mask_chunk)
                if has_para_pair:
                    para_a_emb_grad = self.model.encode_text(para_in_a_chunk, para_mask_a_chunk)
                    para_b_emb_grad = self.model.encode_text(para_in_b_chunk, para_mask_b_chunk)
                if has_img_aug_pair:
                    img_aug_a_emb_grad = self.model.encode_image(img_aug_a_chunk)
                    img_aug_b_emb_grad = self.model.encode_image(img_aug_b_chunk)

            # Build full-batch embeddings: current chunk has gradients, others are cached (detached)
            chunk_start = chunk_idx * self.micro_batch_size
            chunk_end = chunk_start + img_chunk.size(0)

            img_embeds_full = torch.cat([cached_img_embeds[:chunk_start], img_emb_grad, cached_img_embeds[chunk_end:]], dim=0)
            txt_embeds_full = torch.cat([cached_txt_embeds[:chunk_start], txt_emb_grad, cached_txt_embeds[chunk_end:]], dim=0)
            para_a_embeds_full = None
            para_b_embeds_full = None
            if has_para_pair:
                para_a_embeds_full = torch.cat([
                    cached_para_a_embeds_full[:chunk_start],
                    para_a_emb_grad,
                    cached_para_a_embeds_full[chunk_end:],
                ], dim=0)
                para_b_embeds_full = torch.cat([
                    cached_para_b_embeds_full[:chunk_start],
                    para_b_emb_grad,
                    cached_para_b_embeds_full[chunk_end:],
                ], dim=0)
            img_aug_a_embeds_full = None
            img_aug_b_embeds_full = None
            if has_img_aug_pair:
                img_aug_a_embeds_full = torch.cat([
                    cached_img_aug_a_embeds_full[:chunk_start],
                    img_aug_a_emb_grad,
                    cached_img_aug_a_embeds_full[chunk_end:],
                ], dim=0)  # [N, D]
                img_aug_b_embeds_full = torch.cat([
                    cached_img_aug_b_embeds_full[:chunk_start],
                    img_aug_b_emb_grad,
                    cached_img_aug_b_embeds_full[chunk_end:],
                ], dim=0)  # [N, D]

            # Compute loss on full batch (but only current chunk has gradients)
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    loss_dict = self.criterion(
                        img_embeds_full,
                        txt_embeds_full,
                        self.model.clip.logit_scale,
                        img_aug_a_embeds=img_aug_a_embeds_full,
                        img_aug_b_embeds=img_aug_b_embeds_full,
                        txt_aug_a_embeds=para_a_embeds_full,
                        txt_aug_b_embeds=para_b_embeds_full,
                    )
            else:
                loss_dict = self.criterion(
                    img_embeds_full,
                    txt_embeds_full,
                    self.model.clip.logit_scale,
                    img_aug_a_embeds=img_aug_a_embeds_full,
                    img_aug_b_embeds=img_aug_b_embeds_full,
                    txt_aug_a_embeds=para_a_embeds_full,
                    txt_aug_b_embeds=para_b_embeds_full,
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

        # logit_scale participates in every cached full-batch loss, so the loop
        # above accumulates its gradient once per micro-batch. Model-parameter
        # gradients are chunk-local by construction; only this shared scalar
        # needs normalization.
        if getattr(self.model.clip, "logit_scale", None) is not None:
            logit_scale = self.model.clip.logit_scale
            if logit_scale.grad is not None and num_chunks > 1:
                logit_scale.grad.div_(num_chunks)
        
        # Return averaged loss dictionary (for logging)
        # Convert back to tensors for consistency with standard forward
        return {
            key: torch.tensor(value, device=self.device) 
            for key, value in loss_dict_accumulator.items()
        }
    
    def _standard_forward(self, images, input_ids, attention_mask,
                          para_input_ids_a=None, para_attention_mask_a=None,
                          para_input_ids_b=None, para_attention_mask_b=None,
                          image_aug_a=None, image_aug_b=None):
        """Standard forward pass when gradient caching is not needed."""
        w_txt = getattr(self.criterion, 'w_txt', 0)
        has_para_pair = (
            para_input_ids_a is not None
            and para_attention_mask_a is not None
            and para_input_ids_b is not None
            and para_attention_mask_b is not None
            and w_txt > 0
        )
        w_img = getattr(self.criterion, 'w_img', 0)
        has_img_aug_pair = (image_aug_a is not None) and (image_aug_b is not None) and (w_img > 0)
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                img_embeds = self.model.encode_image(images)
                txt_embeds = self.model.encode_text(input_ids, attention_mask)
                para_a_embeds = (
                    self.model.encode_text(para_input_ids_a, para_attention_mask_a)
                    if has_para_pair else None
                )
                para_b_embeds = (
                    self.model.encode_text(para_input_ids_b, para_attention_mask_b)
                    if has_para_pair else None
                )
                img_aug_a_embeds = (
                    self.model.encode_image(image_aug_a) if has_img_aug_pair else None
                )
                img_aug_b_embeds = (
                    self.model.encode_image(image_aug_b) if has_img_aug_pair else None
                )
                loss_dict = self.criterion(
                    img_embeds, txt_embeds,
                    self.model.clip.logit_scale,
                    img_aug_a_embeds=img_aug_a_embeds,
                    img_aug_b_embeds=img_aug_b_embeds,
                    txt_aug_a_embeds=para_a_embeds,
                    txt_aug_b_embeds=para_b_embeds,
                )
        else:
            img_embeds = self.model.encode_image(images)
            txt_embeds = self.model.encode_text(input_ids, attention_mask)
            para_a_embeds = (
                self.model.encode_text(para_input_ids_a, para_attention_mask_a)
                if has_para_pair else None
            )
            para_b_embeds = (
                self.model.encode_text(para_input_ids_b, para_attention_mask_b)
                if has_para_pair else None
            )
            img_aug_a_embeds = (
                self.model.encode_image(image_aug_a) if has_img_aug_pair else None
            )
            img_aug_b_embeds = (
                self.model.encode_image(image_aug_b) if has_img_aug_pair else None
            )
            loss_dict = self.criterion(
                img_embeds, txt_embeds,
                self.model.clip.logit_scale,
                img_aug_a_embeds=img_aug_a_embeds,
                img_aug_b_embeds=img_aug_b_embeds,
                txt_aug_a_embeds=para_a_embeds,
                txt_aug_b_embeds=para_b_embeds,
            )

        return loss_dict
