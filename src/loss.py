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


class SigLIPLoss(nn.Module):
    """
    Pairwise Sigmoid Loss for Language-Image Pre-training (SigLIP).

    Unlike InfoNCE, which uses softmax normalization across the batch, SigLIP treats
    each image-text pair as an independent binary classification:

        z_ij = logit_scale * sim(image_i, text_j) + bias
        y_ij = +1 if i == j (positive pair), -1 otherwise (negative pair)
        L = -mean_ij [ log sigmoid(y_ij * z_ij) ]

    Equivalently, using the identity  log sigmoid(y*z) = log_sigmoid(y*z):
        L = -mean( F.logsigmoid(labels * logits) )

    Key differences from InfoNCE:
    - No softmax denominator: each pair is scored independently.
    - A learnable bias term (initialized to -10) shifts all logits negative at the
      start, so the model is not overconfident on the N^2 - N negatives.
    - Gradients do NOT vanish when the temperature is very sharp (no softmax).

    Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (ICCV 2023).

    Args:
        config (dict): Configuration dict. No keys are read from it; accepts for
                       interface compatibility with SymmetricInfoNCELoss.
    """

    def __init__(self, config):
        super().__init__()
        # Learnable bias initialized to -10 so that sigmoid(-10) ≈ 0 at step 0,
        # preventing the model from being overconfident on the N²-N negatives.
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, img_embeds, txt_embeds, logit_scale, img_aug_embeds=None, txt_aug_embeds=None):
        """
        Compute SigLIP loss.

        Args:
            img_embeds: [N, D] L2-normalized image embeddings
            txt_embeds: [N, D] L2-normalized text embeddings
            logit_scale: scalar nn.Parameter — model.clip.logit_scale (log scale)
            img_aug_embeds: ignored (SigLIP has no intra-modal term)
            txt_aug_embeds: ignored (SigLIP has no intra-modal term)

        Returns:
            dict: Same keys as SymmetricInfoNCELoss:
                - "loss_total": SigLIP loss
                - "loss_inter": same as loss_total
                - "loss_intra_img": zero tensor
                - "loss_intra_txt": zero tensor
        """
        N = img_embeds.shape[0]

        scale = logit_scale.exp().clamp(max=100)  # scalar

        # Full N×N similarity matrix
        sim = torch.matmul(img_embeds, txt_embeds.t())  # [N, N]
        logits = scale * sim + self.bias                # [N, N]

        # Labels: +1 on diagonal (positive pairs), -1 off-diagonal (negative pairs)
        labels = 2 * torch.eye(N, device=img_embeds.device) - 1  # [N, N]

        # Pairwise sigmoid loss (numerically stable via F.logsigmoid)
        loss_total = -F.logsigmoid(labels * logits).mean()  # scalar

        assert not torch.isnan(loss_total), "SigLIP loss is NaN"
        assert not torch.isinf(loss_total), "SigLIP loss is Inf"

        zero = torch.tensor(0.0, device=img_embeds.device)
        return {
            "loss_total": loss_total,
            "loss_inter": loss_total,
            "loss_intra_img": zero,
            "loss_intra_txt": zero,
        }


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
        # Separate weights for Intra-Modal Image and Text consistency
        # Set to 0.0 to disable intra-modal regularization
        self.w_img = config['loss'].get('intra_img_weight', 0.0)
        self.w_txt = config['loss'].get('intra_txt_weight', 0.0)

        self.criterion = nn.CrossEntropyLoss()

    def _compute_contrastive(self, embed_a, embed_b, scale):
        """
        Computes the Symmetric InfoNCE (Information Noise Contrastive Estimation) loss
        between two sets of embeddings a and b.

        The loss is computed symmetrically:
        1.  Loss(a->b): For each element in 'a', find the matching element in 'b'.
        2.  Loss(b->a): For each element in 'b', find the matching element in 'a'.

        This assumes a 1-to-1 correspondence between embed_a[i] and embed_b[i].

        Args:
            embed_a (Tensor): (N, D) Normalized embeddings
            embed_b (Tensor): (N, D) Normalized embeddings
            scale (Tensor): scalar — logit_scale.exp().clamp(max=100)

        Returns:
            torch.Tensor: Scalar loss value (average of a->b and b->a losses).
        """
        # Similarity Matrix: [Batch_Size, Batch_Size]
        # Entry [i, j] = cosine similarity between embed_a[i] and embed_b[j]
        logits = torch.matmul(embed_a, embed_b.t()) * scale  # [N, N]
        
        # Ground Truth Labels: Diagonal indices [0, 1, 2, ..., N-1]
        # Since inputs are aligned (embed_a[i] matches embed_b[i]),
        # the correct match for embed_a[i] is embed_b[i] (index i)
        batch_size = embed_a.shape[0]
        labels = torch.arange(batch_size, device=embed_a.device)
        
        # Symmetric Loss Computation
        # Direction 1: A→B (Row-wise softmax) → "Which embed_b matches embed_a[i]?"
        loss_a2b = self.criterion(logits, labels)
        
        # Direction 2: B→A (Column-wise softmax) → "Which embed_a matches embed_b[j]?"
        loss_b2a = self.criterion(logits.t(), labels)
        
        # Return average of both directions
        return (loss_a2b + loss_b2a) / 2

    def forward(self, img_embeds, txt_embeds, logit_scale, img_aug_embeds=None, txt_aug_embeds=None):
        """
        Computes the complete retrieval loss with optional intra-modal regularization.

        Args:
            img_embeds: [N, D] - L2 normalized image embeddings
            txt_embeds: [N, D] - L2 normalized text embeddings (positives)
            logit_scale: scalar nn.Parameter — model.clip.logit_scale (log scale)
            img_aug_embeds: [N, D] optional - for intra-modal image loss
            txt_aug_embeds: [N, D] optional - for intra-modal text loss

        Returns:
            dict: Dictionary containing total loss and individual components:
                - "loss_total": Final weighted sum
                - "loss_inter": Image-Text contrastive loss
                - "loss_intra_img": Image-Image intra-modal loss
                - "loss_intra_txt": Text-Text intra-modal loss
        """
        scale = logit_scale.exp().clamp(max=100)  # scalar

        # Inter-Modal Loss (Image <-> Text)
        loss_inter = self._compute_contrastive(img_embeds, txt_embeds, scale)

        loss_img = torch.tensor(0.0, device=img_embeds.device)
        if self.w_img > 0 and img_aug_embeds is not None:
            loss_img = self._compute_contrastive(img_embeds, img_aug_embeds, scale)

        loss_txt = torch.tensor(0.0, device=txt_embeds.device)
        if self.w_txt > 0 and txt_aug_embeds is not None:
            loss_txt = self._compute_contrastive(txt_embeds, txt_aug_embeds, scale)

        total_loss = loss_inter + (self.w_img * loss_img) + (self.w_txt * loss_txt)

        return {
            "loss_total": total_loss,
            "loss_inter": loss_inter,
            "loss_intra_img": loss_img,
            "loss_intra_txt": loss_txt
        }


def build_loss(config):
    """Factory function to instantiate the correct loss from config['loss']['type']."""
    loss_type = config.get('loss', {}).get('type', 'infonce').lower()
    if loss_type == 'siglip':
        return SigLIPLoss(config)
    elif loss_type == 'infonce':
        return SymmetricInfoNCELoss(config)
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'. Choose 'infonce' or 'siglip'.")