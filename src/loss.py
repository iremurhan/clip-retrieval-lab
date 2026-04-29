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

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        config (dict): Configuration dict. Reads:
                       - loss.intra_img_weight (default: 0.0)
                       - loss.intra_txt_weight (default: 0.0)
    """

    def __init__(self, config):
        super().__init__()
        # Learnable bias initialized to -10 so that sigmoid(-10) ≈ 0 at step 0,
        # preventing the model from being overconfident on the N²-N negatives.
        self.bias = nn.Parameter(torch.tensor(-10.0))
        self.w_img = config['loss']['intra_img_weight']
        self.w_txt = config['loss']['intra_txt_weight']

    def _compute_siglip(self, embeds_a, embeds_b, logit_scale):
        """Pairwise sigmoid loss between two sets of L2-normalized embeddings."""
        n = embeds_a.shape[0]
        sim = embeds_a @ embeds_b.T                              # [N, N]
        ls = logit_scale.exp().clamp(max=100)                    # scalar
        if ls.item() >= 99.0:
            logger.warning(f"logit_scale.exp()={ls.item():.2f} hit clamp ceiling")
        logits = ls * sim + self.bias                            # [N, N]
        labels = 2 * torch.eye(n, device=embeds_a.device) - 1   # [N, N]
        return -F.logsigmoid(labels * logits).mean()             # scalar

    def _compute_siglip_with_hard_negatives(self, img_embeds, txt_embeds, neg_txt_embeds, logit_scale):
        """
        SigLIP variant with hard negatives.
        Hard negatives are treated as additional negative pairs with label -1.

        img_embeds:     [N, D]
        txt_embeds:     [N, D] — original captions (positives on diagonal)
        neg_txt_embeds: [N, D] — hard negative captions (never positive)
        """
        n = img_embeds.shape[0]
        ls = logit_scale.exp().clamp(max=100)                    # scalar

        # Similarity with original captions: [N, N]
        sim_pos = img_embeds @ txt_embeds.T                      # [N, N]
        logits_pos = ls * sim_pos + self.bias                    # [N, N]
        labels_pos = 2 * torch.eye(n, device=img_embeds.device) - 1  # [N, N]
        loss_pos = -F.logsigmoid(labels_pos * logits_pos).mean() # scalar

        # Similarity with hard negatives: [N, N] — all negative pairs
        sim_neg = img_embeds @ neg_txt_embeds.T                  # [N, N]
        logits_neg = ls * sim_neg + self.bias                    # [N, N]
        labels_neg = -torch.ones(n, n, device=img_embeds.device) # [N, N]
        loss_neg = -F.logsigmoid(labels_neg * logits_neg).mean() # scalar

        return (loss_pos + loss_neg) / 2

    def forward(self, img_embeds, txt_embeds, logit_scale,
                img_aug_a_embeds=None, img_aug_b_embeds=None,
                txt_aug_a_embeds=None, txt_aug_b_embeds=None,
                neg_txt_embeds=None):
        """
        Compute SigLIP loss with optional intra-modal terms (image-image and
        text-text), each computed between two independent augmented or
        paraphrased views of the same sample.

        Args:
            img_embeds: [N, D] L2-normalized image embeddings (inter-modal anchor — clean view).
            txt_embeds: [N, D] L2-normalized text embeddings (inter-modal anchor — original caption).
            logit_scale: scalar nn.Parameter — model.clip.logit_scale (log scale).
            img_aug_a_embeds, img_aug_b_embeds: [N, D] optional — two augmented image
                views for the intra-modal image-image SigLIP pair (L_img_img).
            txt_aug_a_embeds, txt_aug_b_embeds: [N, D] optional — two paraphrase text
                embeddings for the intra-modal text-text SigLIP pair (L_text_text).
            neg_txt_embeds: [N, D] optional — hard-negative captions (B2).

        Returns:
            dict with keys: loss_total, loss_inter, loss_intra_img, loss_intra_txt.
        """
        if neg_txt_embeds is not None:
            loss_inter = self._compute_siglip_with_hard_negatives(
                img_embeds, txt_embeds, neg_txt_embeds, logit_scale)
        else:
            loss_inter = self._compute_siglip(img_embeds, txt_embeds, logit_scale)

        loss_img = torch.tensor(0.0, device=img_embeds.device)
        if self.w_img > 0 and img_aug_a_embeds is not None and img_aug_b_embeds is not None:
            loss_img = self._compute_siglip(img_aug_a_embeds, img_aug_b_embeds, logit_scale)

        loss_txt = torch.tensor(0.0, device=img_embeds.device)
        if self.w_txt > 0 and txt_aug_a_embeds is not None and txt_aug_b_embeds is not None:
            loss_txt = self._compute_siglip(txt_aug_a_embeds, txt_aug_b_embeds, logit_scale)

        loss_total = loss_inter + self.w_img * loss_img + self.w_txt * loss_txt

        assert not torch.isnan(loss_total), "SigLIP loss_total is NaN"
        assert not torch.isinf(loss_total), "SigLIP loss_total is Inf"

        return {
            "loss_total": loss_total,
            "loss_inter": loss_inter,
            "loss_intra_img": loss_img,
            "loss_intra_txt": loss_txt,
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
        self.w_img = config['loss']['intra_img_weight']
        self.w_txt = config['loss']['intra_txt_weight']

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

    def _compute_contrastive_with_hard_negatives(self, img_embeds, txt_embeds, neg_txt_embeds, scale):
        """
        Extends the N×N similarity matrix to N×(2N) by appending hard negatives.
        Hard negatives are additional text negatives — never treated as positives.

        img_embeds:     [N, D]
        txt_embeds:     [N, D] — original captions (positives on diagonal)
        neg_txt_embeds: [N, D] — hard negative captions (never positive)
        scale:          scalar — logit_scale.exp().clamp(max=100)

        Returns scalar loss (i2t uses N×2N, t2i uses N×N only).
        """
        n = img_embeds.shape[0]

        # [N, 2N] similarity matrix: positives in columns [0..N-1], hard negs in [N..2N-1]
        all_txt = torch.cat([txt_embeds, neg_txt_embeds], dim=0)  # [2N, D]
        logits_i2t = torch.matmul(img_embeds, all_txt.t()) * scale  # [N, 2N]

        # Labels: for image i, the positive is text i (index i in [0..N-1])
        labels = torch.arange(n, device=img_embeds.device)

        # i2t: each image finds its caption among 2N candidates
        loss_i2t = self.criterion(logits_i2t, labels)

        # t2i: each original caption finds its image among N candidates
        # Use only the N×N submatrix (hard negs don't have matching images)
        logits_t2i = torch.matmul(txt_embeds, img_embeds.t()) * scale  # [N, N]
        loss_t2i = self.criterion(logits_t2i, labels)

        return (loss_i2t + loss_t2i) / 2

    def forward(self, img_embeds, txt_embeds, logit_scale,
                img_aug_a_embeds=None, img_aug_b_embeds=None,
                txt_aug_a_embeds=None, txt_aug_b_embeds=None,
                neg_txt_embeds=None):
        """
        Compute the complete InfoNCE retrieval loss with optional intra-modal
        regularization (image-image and text-text, each between two independent
        augmented or paraphrased views of the same sample) and optional
        NegCLIP-style hard negatives.

        Args:
            img_embeds: [N, D] L2-normalized image embeddings (inter-modal anchor — clean view).
            txt_embeds: [N, D] L2-normalized text embeddings (inter-modal anchor — original caption).
            logit_scale: scalar nn.Parameter — model.clip.logit_scale (log scale).
            img_aug_a_embeds, img_aug_b_embeds: [N, D] optional — two augmented image
                views for the intra-modal image-image InfoNCE pair (L_img_img).
            txt_aug_a_embeds, txt_aug_b_embeds: [N, D] optional — two paraphrase text
                embeddings for the intra-modal text-text InfoNCE pair (L_text_text).
            neg_txt_embeds: [N, D] optional — hard-negative captions; extends the i2t
                similarity matrix to N×2N (NegCLIP-asymmetric, B2).

        Returns:
            dict: { "loss_total", "loss_inter", "loss_intra_img", "loss_intra_txt" }.
        """
        scale = logit_scale.exp().clamp(max=100)  # scalar
        ls_val = logit_scale.exp().item()
        if ls_val >= 99.0:
            logger.warning(
                f"logit_scale.exp()={ls_val:.2f} has hit the clamp ceiling (100). "
                "This indicates temperature instability."
            )

        # Inter-Modal Loss (Image <-> Text)
        if neg_txt_embeds is not None:
            loss_inter = self._compute_contrastive_with_hard_negatives(
                img_embeds, txt_embeds, neg_txt_embeds, scale)
        else:
            loss_inter = self._compute_contrastive(img_embeds, txt_embeds, scale)

        loss_img = torch.tensor(0.0, device=img_embeds.device)
        if self.w_img > 0 and img_aug_a_embeds is not None and img_aug_b_embeds is not None:
            loss_img = self._compute_contrastive(img_aug_a_embeds, img_aug_b_embeds, scale)

        loss_txt = torch.tensor(0.0, device=txt_embeds.device)
        if self.w_txt > 0 and txt_aug_a_embeds is not None and txt_aug_b_embeds is not None:
            loss_txt = self._compute_contrastive(txt_aug_a_embeds, txt_aug_b_embeds, scale)

        total_loss = loss_inter + (self.w_img * loss_img) + (self.w_txt * loss_txt)

        if torch.isnan(total_loss):
            raise RuntimeError("SymmetricInfoNCELoss is NaN. Check embeddings and logit_scale.")
        if torch.isinf(total_loss):
            raise RuntimeError("SymmetricInfoNCELoss is Inf. Check embeddings and logit_scale.")

        return {
            "loss_total": total_loss,
            "loss_inter": loss_inter,
            "loss_intra_img": loss_img,
            "loss_intra_txt": loss_txt
        }


def build_loss(config):
    """Factory function to instantiate the correct loss from config['loss']['type']."""
    if 'loss' not in config:
        raise KeyError("Config missing 'loss' section")
    for required_key in ['type', 'intra_img_weight', 'intra_txt_weight']:
        if required_key not in config['loss']:
            raise KeyError(
                f"Config missing 'loss.{required_key}'. "
                f"Add it to config_base.yaml."
            )
    loss_type = config['loss']['type'].lower()
    if loss_type == 'siglip':
        return SigLIPLoss(config)
    elif loss_type == 'infonce':
        return SymmetricInfoNCELoss(config)
    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Valid options: 'infonce', 'siglip'"
        )