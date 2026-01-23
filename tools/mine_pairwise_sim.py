#!/usr/bin/env python3
"""
calculate_pairwise_similarity.py
--------------------------------
End-to-End Joint Pairwise Similarity Mining.

Core Logic:
For each image pair (Image A, Image B):
1. Build a 5x5 similarity matrix between their 5 captions each
2. MAX Strategy: Extract the MAXIMUM value from the 5x5 matrix
   - This is the "optimistic" similarity: best caption pair match
   - Result: max_sim[A, B] = max over all 25 caption pairs
3. MIN Strategy: Extract the MINIMUM value from the 5x5 matrix
   - This is the "robust" similarity: worst caption pair match
   - Result: min_sim[A, B] = min over all 25 caption pairs
   - High min-score means robust match (even worst captions agree)

After computing all pairwise similarities:
- For each image, retrieve Top-K neighbors with HIGHEST scores
- MAX mode: Images with highest max-similarity (best matches)
- MIN mode: Images with highest min-similarity (robust matches)

Flow:
1. Load CLIP Model & Dataset (SafeTensors enabled)
2. Extract all Caption Embeddings on-the-fly
3. Free VRAM
4. Perform Block Matrix Multiplication (compute all pairwise similarities)
5. Save final indices/scores (Top-K per image)
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
    logger.info(f"Loading CLIP: {model_name}")
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    except Exception as e:
        logger.warning(f"SafeTensors load failed: {e}. Trying legacy load.")
        model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return model, tokenizer

def compute_all_caption_embeddings(dataset, tokenizer, model, batch_size, device, max_len):
    all_embeds = []
    n_samples = len(dataset.samples)
    logger.info(f"Extracting features for {n_samples} captions...")
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting"):
            end = min(i + batch_size, n_samples)
            batch_caps = [dataset.samples[j]['caption'] for j in range(i, end)]
            inputs = tokenizer(batch_caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            embeds = model.get_text_features(**inputs)
            embeds = F.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())
            
    return torch.cat(all_embeds, dim=0)

def mine_joint_max_min(all_embeds, image_to_indices, image_ids, top_k, device, query_chunk_size=50):
    n_images = len(image_ids)
    dim = all_embeds.shape[1]
    
    logger.info("Reshaping embeddings to [N, 5, Dim]...")
    structured_embeds = torch.zeros(n_images, 5, dim, device=device)
    all_embeds_gpu = all_embeds.to(device)
    
    for i, img_id in enumerate(tqdm(image_ids, desc="Reshaping")):
        indices = image_to_indices[img_id][:5]
        if len(indices) < 5: indices += [indices[0]] * (5 - len(indices))
        structured_embeds[i] = all_embeds_gpu[indices]
    
    del all_embeds_gpu
    torch.cuda.empty_cache()
    
    max_indices_list, max_scores_list = [], []
    min_indices_list, min_scores_list = [], []
    
    keys_flat = structured_embeds.view(-1, dim)
    n_chunks = (n_images + query_chunk_size - 1) // query_chunk_size
    
    logger.info(f"Mining Joint Max/Min (Top-{top_k} Highest Scores)...")
    
    for i in tqdm(range(0, n_images, query_chunk_size), desc="Mining"):
        end = min(i + query_chunk_size, n_images)
        batch_size = end - i
        
        queries_flat = structured_embeds[i:end].view(-1, dim)
        
        # Compute similarity matrix: [B*5, N*5] -> [B, 5, N, 5]
        # full_sim[b, q_cap, k_img, k_cap] = similarity between:
        #   - Query image b's q_cap-th caption
        #   - Key image k_img's k_cap-th caption
        full_sim = queries_flat @ keys_flat.T
        full_sim = full_sim.view(batch_size, 5, n_images, 5)
        
        # --- MAX STRATEGY ---
        # For each image pair (query_img, key_img):
        #   Extract the MAXIMUM similarity from the 5x5 caption matrix
        #   This gives the "optimistic" similarity score
        # Step 1: Max over key image's 5 captions -> [B, 5, N]
        max_sim_img, _ = full_sim.max(dim=3)
        # Step 2: Max over query image's 5 captions -> [B, N]
        # Result: max_sim_img[b, k] = MAX similarity between query image b and key image k
        max_sim_img, _ = max_sim_img.max(dim=1)
        
        # Mask self-similarity (query image with itself)
        for b in range(batch_size):
            max_sim_img[b, i+b] = -float('inf')
        
        # Retrieve Top-K images with highest MAX similarity scores
        b_max_scores, b_max_indices = max_sim_img.topk(top_k, dim=1)
        max_indices_list.append(b_max_indices.cpu())
        max_scores_list.append(b_max_scores.cpu())
        
        # --- MIN STRATEGY ---
        # For each image pair (query_img, key_img):
        #   Extract the MINIMUM similarity from the 5x5 caption matrix
        #   This gives the "robust" similarity score (even worst captions agree)
        # Step 1: Min over key image's 5 captions -> [B, 5, N]
        min_sim_img, _ = full_sim.min(dim=3)
        # Step 2: Min over query image's 5 captions -> [B, N]
        # Result: min_sim_img[b, k] = MIN similarity between query image b and key image k
        min_sim_img, _ = min_sim_img.min(dim=1)
        
        # Mask self-similarity (query image with itself)
        for b in range(batch_size):
            min_sim_img[b, i+b] = -float('inf')
        
        # Retrieve Top-K images with highest MIN similarity scores
        # (High min-score means robust match: even worst caption pairs have high similarity)
        b_min_scores, b_min_indices = min_sim_img.topk(top_k, dim=1)
        
        min_indices_list.append(b_min_indices.cpu())
        min_scores_list.append(b_min_scores.cpu())
        
        if (i // query_chunk_size + 1) % 10 == 0:
            wandb.log({"mining_progress": (i / n_images) * 100})
            
    return (
        torch.cat(max_indices_list), torch.cat(max_scores_list),
        torch.cat(min_indices_list), torch.cat(min_scores_list)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--extraction_batch_size', type=int, default=512)
    parser.add_argument('--query_chunk_size', type=int, default=50)
    args = parser.parse_args()
    
    wandb.init(project="pairwise-mining", job_type="joint_mining_fly")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config) as f: config = yaml.safe_load(f)
    model, tokenizer = load_clip_model(config['model']['image_model_name'], device)
    
    dataset = CocoImageDataset(
        images_root_path=config['data']['images_path'],
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer, split='train', transform=None, intra_modal_aug=False
    )
    
    image_to_indices = defaultdict(list)
    image_ids_ordered = []
    seen = set()
    for idx, sample in enumerate(dataset.samples):
        iid = sample['image_id']
        image_to_indices[iid].append(idx)
        if iid not in seen:
            image_ids_ordered.append(iid)
            seen.add(iid)
            
    all_embeds = compute_all_caption_embeddings(dataset, tokenizer, model, args.extraction_batch_size, device, 77)
    del model
    torch.cuda.empty_cache()
    
    max_idxs, max_scores, min_idxs, min_scores = mine_joint_max_min(
        all_embeds, image_to_indices, image_ids_ordered, args.top_k, device, args.query_chunk_size
    )
    
    logger.info("Mapping results to caption level...")
    n_caps = len(dataset.samples)
    out_max_indices = torch.zeros(n_caps, args.top_k, dtype=torch.long)
    out_max_scores = torch.zeros(n_caps, args.top_k, dtype=torch.float32)
    out_min_indices = torch.zeros(n_caps, args.top_k, dtype=torch.long)
    out_min_scores = torch.zeros(n_caps, args.top_k, dtype=torch.float32)

    img_idx_to_first_cap = {i: image_to_indices[iid][0] for i, iid in enumerate(image_ids_ordered)}
    
    for i, iid in enumerate(tqdm(image_ids_ordered, desc="Mapping")):
        max_neigh = torch.tensor([img_idx_to_first_cap[n.item()] for n in max_idxs[i]])
        min_neigh = torch.tensor([img_idx_to_first_cap[n.item()] for n in min_idxs[i]])
        
        for cap_idx in image_to_indices[iid]:
            out_max_indices[cap_idx] = max_neigh
            out_max_scores[cap_idx] = max_scores[i]
            out_min_indices[cap_idx] = min_neigh
            out_min_scores[cap_idx] = min_scores[i]

    save_dict = {
        "max_indices": out_max_indices, "max_scores": out_max_scores,
        "min_indices": out_min_indices, "min_scores": out_min_scores
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(save_dict, args.output_path)
    logger.info(f"Saved to {args.output_path}")
    wandb.finish()

if __name__ == "__main__":
    main()