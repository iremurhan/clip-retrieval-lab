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
        logger.warning(f"SafeTensors load failed: {e}")
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

    logger.info(f"Extracting features for {len(dataset)} captions...")
    
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
    
    logger.info(f"Mining Joint Max/Min... (Query Chunk: {query_chunk_size})")
    
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
        
        min_indices_list.append(b_min_indices.cpu())
        min_scores_list.append(b_min_scores.cpu())
        
        # WandB Log
        chunk_idx = i // query_chunk_size
        if (chunk_idx + 1) % 10 == 0:
            wandb.log({"mining_progress": (chunk_idx + 1) / n_chunks * 100})
    
    return (
        torch.cat(max_indices_list), torch.cat(max_scores_list),
        torch.cat(min_indices_list), torch.cat(min_scores_list)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512, help="Extraction batch size (deprecated, use --extraction_batch_size)")
    parser.add_argument('--extraction_batch_size', type=int, default=None, help="Batch size for feature extraction")
    parser.add_argument('--query_chunk_size', type=int, default=50, help="Mining chunk size")
    args = parser.parse_args()
    
    # Handle backward compatibility: if extraction_batch_size not provided, use batch_size
    if args.extraction_batch_size is None:
        args.extraction_batch_size = args.batch_size
    
    # Initialize WandB
    wandb.init(project="pairwise-mining", job_type="joint_mining_fly")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on {device}...")
    
    # 1. Load Config & Dataset (Modified to use SafeTensors)
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f: 
        config = yaml.safe_load(f)
    
    logger.info("Loading CLIP model and dataset...")
    model, tokenizer = load_clip_model(config['model']['image_model_name'], device)
    
    dataset = CocoImageDataset(
        images_root_path=config['data']['images_path'],
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer,
        split='train',
        transform=None,
        intra_modal_aug=False
    )
    logger.info(f"Dataset loaded: {len(dataset.samples)} samples")
    
    # 2. Build Index Map
    logger.info("Building image-to-caption index map...")
    image_to_indices = defaultdict(list)
    image_ids_ordered = []
    seen = set()
    for idx, sample in enumerate(dataset.samples):
        iid = sample['image_id']
        image_to_indices[iid].append(idx)
        if iid not in seen:
            image_ids_ordered.append(iid)
            seen.add(iid)
    
    n_images = len(image_ids_ordered)
    n_captions = len(dataset.samples)
    logger.info(f"  Unique images: {n_images:,}")
    logger.info(f"  Total captions: {n_captions:,}")

    # 3. EXTRACT FEATURES (On-the-fly)
    logger.info("Phase 2: Extracting caption embeddings...")
    all_embeds = compute_all_caption_embeddings(dataset, tokenizer, model, args.extraction_batch_size, device, 77)
    logger.info(f"Extracted embeddings shape: {all_embeds.shape}")
    
    # 4. FREE VRAM (Critical!)
    logger.info("Deleting CLIP model to free VRAM for mining...")
    del model
    torch.cuda.empty_cache()
    
    # 5. MINE JOINT MAX/MIN
    logger.info("Phase 4: Joint MAX/MIN mining...")
    max_idxs, max_scores, min_idxs, min_scores = mine_joint_max_min(
        all_embeds, image_to_indices, image_ids_ordered, args.top_k, device, args.query_chunk_size
    )
    logger.info(f"Mining complete. MAX shape: {max_idxs.shape}, MIN shape: {min_idxs.shape}")
    
    # 6. SAVE
    logger.info("Phase 5: Mapping results to caption level and saving...")
    n_caps = len(dataset.samples)
    
    # Output containers
    out_max_indices = torch.zeros(n_caps, args.top_k, dtype=torch.long)
    out_max_scores = torch.zeros(n_caps, args.top_k, dtype=torch.float32)
    out_min_indices = torch.zeros(n_caps, args.top_k, dtype=torch.long)
    out_min_scores = torch.zeros(n_caps, args.top_k, dtype=torch.float32)

    # Helper: Image Index -> Representative Caption Index
    img_idx_to_first_cap = {i: image_to_indices[iid][0] for i, iid in enumerate(image_ids_ordered)}
    
    for i, iid in enumerate(tqdm(image_ids_ordered, desc="Final Mapping")):
        # Neighbor Img indices -> Neighbor Cap indices
        max_neigh_caps = torch.tensor([img_idx_to_first_cap[n.item()] for n in max_idxs[i]])
        min_neigh_caps = torch.tensor([img_idx_to_first_cap[n.item()] for n in min_idxs[i]])
        
        # Broadcast to all captions of this image
        for cap_idx in image_to_indices[iid]:
            out_max_indices[cap_idx] = max_neigh_caps
            out_max_scores[cap_idx] = max_scores[i]
            out_min_indices[cap_idx] = min_neigh_caps
            out_min_scores[cap_idx] = min_scores[i]

    save_dict = {
        "max_indices": out_max_indices, "max_scores": out_max_scores,
        "min_indices": out_min_indices, "min_scores": out_min_scores
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logger.info(f"Saving to {args.output_path}")
    torch.save(save_dict, args.output_path)
    
    # Log final statistics
    file_size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    logger.info("=" * 60)
    logger.info("MINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total images: {n_images:,}")
    logger.info(f"Total captions: {n_caps:,}")
    logger.info(f"Top-K per image: {args.top_k}")
    logger.info(f"MAX - Score range: [{max_scores.min():.4f}, {max_scores.max():.4f}], Mean: {max_scores.mean():.4f}")
    logger.info(f"MIN - Score range: [{min_scores.min():.4f}, {min_scores.max():.4f}], Mean: {min_scores.mean():.4f}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info("=" * 60)
    
    wandb.log({
        "output_file_size_mb": file_size_mb,
        "max_score_min": max_scores.min().item(),
        "max_score_max": max_scores.max().item(),
        "max_score_mean": max_scores.mean().item(),
        "min_score_min": min_scores.min().item(),
        "min_score_max": min_scores.max().item(),
        "min_score_mean": min_scores.mean().item()
    })
    
    wandb.finish()
    logger.info("Done!")

if __name__ == "__main__":
    main()