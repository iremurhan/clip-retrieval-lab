"""
src/loss.py
-----------
Implements the Symmetric InfoNCE (Information Noise Contrastive Estimation) loss,
also known as Contrastive Cross-Entropy Loss, standardized by OpenAI's CLIP model.

This loss functions as a bi-directional contrastive objective that maximizes 
the cosine similarity between matched image-text pairs while minimizing 
similarity with all other mismatched pairs in the batch (in-batch negatives).

Core Mechanism:
    For each image in a batch of size N, the model attempts to identify the correct 
    matching text among the N texts in the batch. The remaining N-1 texts serve as 
    "negative examples" (in-batch negatives). This process is performed symmetrically 
    in both directions: Image→Text and Text→Image.

    The loss encourages the model to:
    1. Maximize similarity between aligned pairs (image[i] ↔ text[i])
    2. Minimize similarity between misaligned pairs (image[i] ↔ text[j], i≠j)

Additionally supports optional intra-modal consistency terms that enforce structural
preservation within each modality (image↔augmented_image, text↔augmented_text).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """
    Computes the Symmetric InfoNCE (Contrastive Cross-Entropy) Loss with optional 
    intra-modal consistency regularization.
    
    Mathematical Formulation:
    
    1. Inter-Modal Loss (Main Objective):
       - Compute cosine similarity matrix S[i,j] = cos(image[i], text[j]) / τ
       - For Image→Text: CrossEntropy(S[i,:], label=i) for each image i
       - For Text→Image: CrossEntropy(S[:,j], label=j) for each text j
       - Loss_Inter = (Loss_I2T + Loss_T2I) / 2
    
    2. Intra-Modal Loss (Optional Regularization):
       - Image Intra-Modal: Contrastive loss between img_embeds and img_aug_embeds
       - Text Intra-Modal: Contrastive loss between txt_embeds and txt_aug_embeds
       - Enforces consistency between original and augmented views within each modality
    
    Total Loss = Loss_Inter + w_img * Loss_Img_Intra + w_txt * Loss_Txt_Intra
    
    where w_img and w_txt are configurable weights (default: 0.0, disabled).
    
    Args:
        config (dict): Configuration dictionary containing:
            - loss.temperature: Temperature scaling parameter τ (controls softmax sharpness)
            - loss.intra_img_weight: Weight for image intra-modal loss (default: 0.0)
            - loss.intra_txt_weight: Weight for text intra-modal loss (default: 0.0)
            
    Returns:
        dict: A dictionary with 'loss_total', 'loss_inter', 'loss_intra_img', 'loss_intra_txt'.
    """
    
    def __init__(self, config):
        super().__init__()
        # Learnable temperature for L_i2t (cross-modal loss); mirrors CLIP's logit_scale usage
        self.temperature = config['loss']['temperature']
        # Weights for intra-modal losses; set to 0.0 to disable
        self.w_img = config['loss'].get('intra_img_weight', 0.0)
        self.w_txt = config['loss'].get('intra_txt_weight', 0.0)
        # Fixed temperature for intra-modal losses (τ=0.07, not updated by optimizer)
        self.register_buffer("tau_intra", torch.tensor(0.07))
        self.criterion = nn.CrossEntropyLoss()

    def _compute_contrastive(self, embed_a, embed_b, temperature):
        """
        Computes the Symmetric InfoNCE loss between two sets of embeddings a and b.

        Args:
            embed_a (Tensor): (N, D) Normalized embeddings
            embed_b (Tensor): (N, D) Normalized embeddings
            temperature: scalar float or 0-dim Tensor — controls softmax sharpness

        Returns:
            torch.Tensor: Scalar loss value (average of a->b and b->a losses).
        """
        # Similarity Matrix: [N, N]; entry [i,j] = cosine similarity embed_a[i] · embed_b[j] / τ
        logits = torch.matmul(embed_a, embed_b.t()) / temperature
        batch_size = embed_a.shape[0]
        labels = torch.arange(batch_size, device=embed_a.device)
        loss_a2b = self.criterion(logits, labels)    # A→B
        loss_b2a = self.criterion(logits.t(), labels)  # B→A
        return (loss_a2b + loss_b2a) / 2

    def forward(self, F_vit, F_text, F_image_norm, F_text_proj_norm, F_image_norm_aug=None):
        """
        Tri-loss topological routing.

        Args:
            F_vit: [N, D] - Branch A image (ViT CLS, L2-normalized)
            F_text: [N, D] - Branch A text (encode_text, L2-normalized)
            F_image_norm: [N, D] - Branch B image (cross-attn pooled orig, L2-normalized)
            F_text_proj_norm: [N, D] - Branch C text (q_proj projected, L2-normalized)
            F_image_norm_aug: [N, D] optional - Branch B aug image for L_i2i

        Returns:
            dict: {loss_total, loss_i2t, loss_i2i, loss_t2t}
        """
        # L_i2t: ViT CLS image vs standard text — cross-modal (learnable τ)
        loss_i2t = self._compute_contrastive(F_vit, F_text, self.temperature)

        # L_i2i: cross-attn pooled orig vs cross-attn pooled aug — image intra-modal (fixed τ)
        loss_i2i = torch.tensor(0.0, device=F_vit.device)
        if self.w_img > 0 and F_image_norm_aug is not None:
            loss_i2i = self._compute_contrastive(F_image_norm, F_image_norm_aug, self.tau_intra)

        # L_t2t: standard text vs W_text projected text CLS — text intra-modal (fixed τ)
        loss_t2t = torch.tensor(0.0, device=F_vit.device)
        if self.w_txt > 0:
            loss_t2t = self._compute_contrastive(F_text, F_text_proj_norm, self.tau_intra)

        loss_total = loss_i2t + (self.w_img * loss_i2i) + (self.w_txt * loss_t2t)

        return {
            "loss_total": loss_total,
            "loss_i2t": loss_i2t,
            "loss_i2i": loss_i2i,
            "loss_t2t": loss_t2t,
        }