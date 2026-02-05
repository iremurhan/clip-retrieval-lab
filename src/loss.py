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
    """
    
    def __init__(self, config):
        super().__init__()
        # Temperature scaling parameter (controls the sharpness of the softmax)
        # Lower temperature → sharper distribution → harder negatives
        # Higher temperature → smoother distribution → softer negatives
        # Typical values: 0.07 (CLIP default) to 0.1 for fine-tuning
        self.temperature = config['loss']['temperature']
        
        # Separate weights for Intra-Modal Image and Text consistency
        # Set to 0.0 to disable intra-modal regularization
        self.w_img = config['loss'].get('intra_img_weight', 0.0)
        self.w_txt = config['loss'].get('intra_txt_weight', 0.0)
        
        self.criterion = nn.CrossEntropyLoss()

    def compute_symmetric_infonce_loss(self, embed_a, embed_b):
        """
        Computes Symmetric InfoNCE loss between two sets of embeddings.
        
        For a batch of size N, this function:
        1. Computes similarity matrix S[i,j] = cos(embed_a[i], embed_b[j]) / τ
        2. Applies Cross-Entropy loss in both directions:
           - Direction A→B: For each embed_a[i], predict correct embed_b[i]
           - Direction B→A: For each embed_b[j], predict correct embed_a[j]
        3. Returns the average of both directions
        
        Args:
            embed_a (Tensor): [Batch_Size, Embed_Dim] - L2 normalized embeddings
            embed_b (Tensor): [Batch_Size, Embed_Dim] - L2 normalized embeddings
                              (aligned: embed_a[i] matches embed_b[i])
        
        Returns:
            Tensor: Scalar symmetric contrastive loss value
        """
        # Similarity Matrix: [Batch_Size, Batch_Size]
        # Entry [i, j] = cosine similarity between embed_a[i] and embed_b[j]
        logits = torch.matmul(embed_a, embed_b.t()) / self.temperature
        
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

    def forward(self, img_embeds, txt_embeds, img_aug_embeds=None, txt_aug_embeds=None, neg_txt_embeds=None):
        """
        Computes the complete retrieval loss with optional intra-modal regularization and hard negatives.
        
        Args:
            img_embeds: [N, D] - L2 normalized image embeddings
            txt_embeds: [N, D] - L2 normalized text embeddings (positives)
            img_aug_embeds: [N, D] optional - for intra-modal image loss
            txt_aug_embeds: [N, D] optional - for intra-modal text loss
            neg_txt_embeds: [N, D] optional - hard negative text embeddings (one per sample)
        
        Returns:
            Tensor: Scalar total loss value
        """
        # 1. Inter-Modal Loss
        if neg_txt_embeds is not None:
            # Image -> (positive + negative) texts: all_texts = [txt_embeds; neg_txt_embeds] -> [2N, D]
            all_texts = torch.cat([txt_embeds, neg_txt_embeds], dim=0)
            logits_per_image = torch.matmul(img_embeds, all_texts.t()) / self.temperature
            batch_size = img_embeds.shape[0]
            labels = torch.arange(batch_size, device=img_embeds.device)
            loss_i2t = self.criterion(logits_per_image, labels)
            # Text -> Image: standard symmetric
            logits_per_text = torch.matmul(txt_embeds, img_embeds.t()) / self.temperature
            loss_t2i = self.criterion(logits_per_text, labels)
            loss_inter = (loss_i2t + loss_t2i) / 2
        else:
            loss_inter = self.compute_symmetric_infonce_loss(img_embeds, txt_embeds)

        # 2. Intra-Modal Loss (unchanged)
        loss_img = 0.0
        if self.w_img > 0 and img_aug_embeds is not None:
            loss_img = self.compute_symmetric_infonce_loss(img_embeds, img_aug_embeds)
        loss_txt = 0.0
        if self.w_txt > 0 and txt_aug_embeds is not None:
            loss_txt = self.compute_symmetric_infonce_loss(txt_embeds, txt_aug_embeds)

        return loss_inter + (self.w_img * loss_img) + (self.w_txt * loss_txt)