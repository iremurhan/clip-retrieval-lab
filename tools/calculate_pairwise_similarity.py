#!/usr/bin/env python3
"""
calculate_pairwise_similarity.py
--------------------------------
End-to-End Joint Pairwise Similarity Mining.

Flow:
1. Load CLIP Model & Dataset (Using SafeTensors to bypass CVE check).
2. Extract all Caption Embeddings on-the-fly (RAM only).
3. DELETE CLIP Model to free VRAM.
4. Perform Block Matrix Multiplication (Joint Max/Min).
5. Save only the final indices/scores.

Usage:
    python tools/calculate_pairwise_similarity.py \
        --config config.yaml \
        --output_path datasets/coco/pairwise_mining_joint.pt \
        --top_k 50
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import os
import sys
import logging
import wandb
from collections import defaultdict
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CocoImageDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clip_model(model_name, device):
    """
    Loads CLIP model using SafeTensors to avoid PyTorch version vulnerability error.
    """
    logger.info(f"Loading CLIP: {model_name}")
    
    try:
        # CRITICAL FIX: use_safetensors=True bypasses the PyTorch < 2.6 security check
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    except Exception as e:
        logger.warning(f"⚠️ SafeTensors load failed: {e}")
        logger.warning("Attempting legacy load (This may fail if PyTorch < 2.6)...")
        model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(device).eval()

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return model, tokenizer

def compute_all_caption_embeddings(dataset, tokenizer, model, batch_size, device, max_len):
    """
    Computes embeddings for ALL captions on-the-fly.
    Returns: [N_total_caps, Dim] Tensor (on CPU to save VRAM for mining).
    """
    all_embeds = []
    
    # Custom collate because dataset returns dicts
    def collate_fn(batch):
        return [b['caption'] for b in batch]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    logger.info(f"🚀 Extracting features for {len(dataset)} captions...")
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
        for batch_caps in tqdm(loader, desc="Extracting"):
            inputs = tokenizer(batch_caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            embeds = model.get_text_features(**inputs)
            embeds = F.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu()) # Keep on CPU until needed

    return torch.cat(all_embeds, dim=0)

def mine_joint_max_min(all_embeds, image_to_indices, image_ids, top_k, device, query_chunk_size=50):
    """
    Mines both MAX and MIN pairwise similarities.
    """
    n_images = len(image_ids)
    dim = all_embeds.shape[1]
    
    logger.info("Reshaping embeddings to [N, 5, Dim] for vectorized mining...")
    structured_embeds = torch.zeros(n_images, 5, dim, device=device)
    
    # Move raw to GPU for fast gather
    all_embeds_gpu = all_embeds.to(device)
    
    for i, img_id in enumerate(tqdm(image_ids, desc="Reshaping")):
        indices = image_to_indices[img_id]
        indices = indices[:5] 
        if len(indices) < 5:
            indices = indices + [indices[0]] * (5 - len(indices))
        structured_embeds[i] = all_embeds_gpu[indices]
    
    del all_embeds_gpu
    torch.cuda.empty_cache()
    
    max_indices_list, max_scores_list = [], []
    min_indices_list, min_scores_list = [], []
    
    logger.info(f"⛏️ Mining Joint Max/Min... (Query Chunk: {query_chunk_size})")
    
    # Keys: All images [N, 5, Dim]
    keys_flat = structured_embeds.view(-1, dim)
    
    n_chunks = (n_images + query_chunk_size - 1) // query_chunk_size
    
    for i in tqdm(range(0, n_images, query_chunk_size), desc="Mining"):
        end = min(i + query_chunk_size, n_images)
        batch_size = end - i
        
        queries = structured_embeds[i:end]
        queries_flat = queries.view(-1, dim) 
        
        # --- THE HEAVY LIFTING ---
        full_sim = queries_flat @ keys_flat.T 
        full_sim = full_sim.view(batch_size, 5, n_images, 5)
        
        # --- MAX ---
        max_sim_img, _ = full_sim.max(dim=3)
        max_sim_img, _ = max_sim_img.max(dim=1)
        for b in range(batch_size): max_sim_img[b, i+b] = -float('inf')
        
        b_max_scores, b_max_indices = max_sim_img.topk(top_k, dim=1)
        max_indices_list.append(b_max_indices.cpu())
        max_scores_list.append(b_max_scores.cpu())
        
        # --- MIN ---
        min_sim_img, _ = full_sim.min(dim=3)
        min_sim_img, _ = min_sim_img.min(dim=1)
        for b in range(batch_size): min_sim_img[b, i+b] = float('inf')
        
        b_min_scores, b_min_indices = (-min_sim_img).topk(top_k, dim=1)
        b_min_scores = -b_min_scores
        
        min_indices_list.append(