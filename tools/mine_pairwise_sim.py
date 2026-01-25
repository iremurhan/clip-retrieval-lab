#!/usr/bin/env python3
"""
mine_pairwise_sim.py
--------------------
Unified Mining Script for Knowledge Distillation.

This script supports THREE mining modalities:

1. VISUAL (Image-Image via Vision Embeddings)
   - Input: Unique Images (~113k)
   - Logic: CLIP Vision Encoder -> Cosine Similarity
   - Output: ID-Based mapping (mode: id_mapping)
   - File: datasets/{dataset}/pairwise_similarities/mining_image_visual.pt

2. CONSENSUS (Image-Image via Caption Agreement)
   - Input: All Captions grouped by Image
   - Logic: CLIP Text Encoder -> Reshape [N, 5, D] -> 5x5 Matrix -> Mean
   - Output: ID-Based mapping (mode: id_mapping)
   - File: datasets/{dataset}/pairwise_similarities/mining_image_consensus.pt

3. CAPTION (Text-Text Direct Similarity)
   - Input: All Captions (~565k)
   - Logic: CLIP Text Encoder -> Direct Cosine Similarity (chunked)
   - Output: Index-Based mapping (mode: index_mapping)
   - File: datasets/{dataset}/pairwise_similarities/mining_text.pt

Usage:
    # Image-Image mining (visual features)
    python tools/mine_pairwise_sim.py --modality visual --config config.yaml
    
    # Image-Image mining (caption consensus)
    python tools/mine_pairwise_sim.py --modality consensus --config config.yaml
    
    # Text-Text mining (replaces mining_targets.py)
    python tools/mine_pairwise_sim.py --modality caption --config config.yaml
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
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from tqdm import tqdm
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CocoImageDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def load_clip_model(model_name: str, device: torch.device):
    """Load CLIP model with SafeTensors fallback."""
    logger.info(f"Loading CLIP: {model_name}")
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    except Exception as e:
        logger.warning(f"SafeTensors load failed: {e}. Trying legacy load.")
        model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(device).eval()
    
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model, tokenizer, processor


def compute_text_embeddings(dataset, tokenizer, model, batch_size: int, device: torch.device, max_len: int = 77):
    """Extract L2-normalized text embeddings for all captions."""
    all_embeds = []
    n_samples = len(dataset.samples)
    logger.info(f"Extracting text features for {n_samples:,} captions...")
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
        for i in tqdm(range(0, n_samples, batch_size), desc="Text Extraction"):
            end = min(i + batch_size, n_samples)
            batch_caps = [dataset.samples[j]['caption'] for j in range(i, end)]
            inputs = tokenizer(batch_caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            embeds = model.get_text_features(**inputs)
            embeds = F.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())
    
    return torch.cat(all_embeds, dim=0)


def compute_image_embeddings(dataset, processor, model, batch_size: int, device: torch.device, image_ids: list):
    """Extract L2-normalized image embeddings for unique images."""
    all_embeds = []
    n_images = len(image_ids)
    
    # Build image_id -> first sample index mapping
    image_id_to_sample = {}
    for idx, sample in enumerate(dataset.samples):
        iid = sample['image_id']
        if iid not in image_id_to_sample:
            image_id_to_sample[iid] = idx
    
    logger.info(f"Extracting image features for {n_images:,} unique images...")
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
        for i in tqdm(range(0, n_images, batch_size), desc="Image Extraction"):
            end = min(i + batch_size, n_images)
            batch_images = []
            
            for img_id in image_ids[i:end]:
                sample_idx = image_id_to_sample[img_id]
                sample = dataset.samples[sample_idx]
                image_path = os.path.join(dataset.images_root_path, sample['filepath'], sample['filename'])
                image = Image.open(image_path).convert('RGB')
                batch_images.append(image)
            
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            embeds = model.get_image_features(**inputs)
            embeds = F.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())
    
    return torch.cat(all_embeds, dim=0)


# =============================================================================
# MODE 1: VISUAL (Image-Image via Vision Embeddings)
# =============================================================================

def mine_visual(model, processor, dataset, image_ids: list, top_k: int, device: torch.device, 
                batch_size: int = 256, query_chunk_size: int = 100):
    """
    Mine Image-Image similarity using CLIP vision embeddings.
    
    Returns ID-based mapping format.
    """
    logger.info("=" * 60)
    logger.info("MODE: VISUAL (Image-Image via Vision Embeddings)")
    logger.info("=" * 60)
    
    # Extract image embeddings
    image_embeds = compute_image_embeddings(dataset, processor, model, batch_size, device, image_ids)
    image_embeds_gpu = image_embeds.to(device)
    
    n_images = len(image_ids)
    logger.info(f"Mining Top-{top_k} neighbors for {n_images:,} images...")
    
    indices_list, scores_list = [], []
    
    for i in tqdm(range(0, n_images, query_chunk_size), desc="Mining Visual"):
        end = min(i + query_chunk_size, n_images)
        queries = image_embeds_gpu[i:end]
        
        # Compute similarity
        sims = queries @ image_embeds_gpu.T  # [chunk, N]
        
        # Mask self-similarity
        for b in range(end - i):
            sims[b, i + b] = -float('inf')
        
        # Get top-k
        top_scores, top_indices = sims.topk(top_k, dim=1)
        indices_list.append(top_indices.cpu())
        scores_list.append(top_scores.cpu())
    
    indices = torch.cat(indices_list, dim=0)
    scores = torch.cat(scores_list, dim=0)
    
    return {
        "mode": "id_mapping",
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "indices": indices,
        "scores": scores
    }


# =============================================================================
# MODE 2: CONSENSUS (Image-Image via Caption Agreement)
# =============================================================================

def mine_consensus(all_text_embeds, image_to_indices, image_ids: list, top_k: int, 
                   device: torch.device, query_chunk_size: int = 50):
    """
    Mine Image-Image similarity using caption consensus (5x5 matrix mean).
    
    Returns ID-based mapping format.
    """
    logger.info("=" * 60)
    logger.info("MODE: CONSENSUS (Image-Image via Caption Agreement)")
    logger.info("=" * 60)
    
    n_images = len(image_ids)
    dim = all_text_embeds.shape[1]
    
    # Reshape embeddings to [N_images, 5, dim]
    logger.info("Reshaping embeddings to [N_images, 5, Dim]...")
    structured_embeds = torch.zeros(n_images, 5, dim, device=device)
    all_embeds_gpu = all_text_embeds.to(device)
    
    for i, img_id in enumerate(tqdm(image_ids, desc="Reshaping")):
        indices = image_to_indices[img_id][:5]
        if len(indices) < 5:
            indices = indices + [indices[0]] * (5 - len(indices))
        structured_embeds[i] = all_embeds_gpu[indices]
    
    del all_embeds_gpu
    torch.cuda.empty_cache()
    
    keys_flat = structured_embeds.view(-1, dim)
    
    logger.info(f"Mining Top-{top_k} neighbors using MEAN consensus...")
    
    indices_list, scores_list = [], []
    
    for i in tqdm(range(0, n_images, query_chunk_size), desc="Mining Consensus"):
        end = min(i + query_chunk_size, n_images)
        batch_size = end - i
        
        queries_flat = structured_embeds[i:end].view(-1, dim)
        
        # Compute 5x5 similarity matrices: [B*5, N*5] -> [B, 5, N, 5]
        full_sim = queries_flat @ keys_flat.T
        full_sim = full_sim.view(batch_size, 5, n_images, 5)
        
        # MEAN aggregation: average over all 25 caption pairs
        mean_sim = full_sim.mean(dim=(1, 3))  # [B, N]
        
        # Mask self-similarity
        for b in range(batch_size):
            mean_sim[b, i + b] = -float('inf')
        
        # Get top-k
        top_scores, top_indices = mean_sim.topk(top_k, dim=1)
        indices_list.append(top_indices.cpu())
        scores_list.append(top_scores.cpu())
    
    indices = torch.cat(indices_list, dim=0)
    scores = torch.cat(scores_list, dim=0)
    
    return {
        "mode": "id_mapping",
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "indices": indices,
        "scores": scores
    }


# =============================================================================
# MODE 3: CAPTION (Text-Text Direct Similarity)
# =============================================================================

def mine_caption(all_text_embeds, top_k: int, device: torch.device, query_chunk_size: int = 1000):
    """
    Mine Text-Text similarity using direct cosine similarity.
    
    This replaces the legacy mining_targets.py script.
    Returns index-based mapping format (1-to-1 with dataset).
    """
    logger.info("=" * 60)
    logger.info("MODE: CAPTION (Text-Text Direct Similarity)")
    logger.info("=" * 60)
    
    n_samples = all_text_embeds.shape[0]
    all_embeds_gpu = all_text_embeds.to(device)
    
    logger.info(f"Mining Top-{top_k} neighbors for {n_samples:,} captions...")
    
    indices_list, scores_list = [], []
    
    for i in tqdm(range(0, n_samples, query_chunk_size), desc="Mining Captions"):
        end = min(i + query_chunk_size, n_samples)
        queries = all_embeds_gpu[i:end]
        
        # Compute similarity against all embeddings
        sims = queries @ all_embeds_gpu.T  # [chunk, N]
        
        # Mask ONLY self-similarity (keep same-image captions as valid positives)
        for b in range(end - i):
            sims[b, i + b] = -float('inf')
        
        # Get top-k
        top_scores, top_indices = sims.topk(top_k, dim=1)
        indices_list.append(top_indices.cpu())
        scores_list.append(top_scores.float().cpu())
    
    indices = torch.cat(indices_list, dim=0)
    scores = torch.cat(scores_list, dim=0)
    
    return {
        "mode": "index_mapping",
        "indices": indices,
        "scores": scores
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Mining Script for Knowledge Distillation')
    parser.add_argument('--modality', type=str, required=True, choices=['visual', 'consensus', 'caption'],
                        help='Mining modality: visual (image-image), consensus (caption agreement), caption (text-text)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: datasets/coco)')
    parser.add_argument('--top_k', type=int, default=50, help='Number of neighbors to mine')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for embedding extraction')
    parser.add_argument('--query_chunk_size', type=int, default=100, help='Query chunk size for mining')
    args = parser.parse_args()
    
    # Device setup
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU (will be slow).")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Determine output directory: datasets/{dataset}/pairwise_similarities/
    base_data_dir = args.output_dir or config['data']['images_path']
    output_dir = os.path.join(base_data_dir, 'pairwise_similarities')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Output filename based on modality
    output_filenames = {
        'visual': 'mining_image_visual.pt',
        'consensus': 'mining_image_consensus.pt',
        'caption': 'mining_text.pt'
    }
    output_path = os.path.join(output_dir, output_filenames[args.modality])
    
    # Initialize wandb
    wandb_project = config.get('logging', {}).get('wandb_project', 'mining')
    wandb.init(project=wandb_project, job_type=f"mining_{args.modality}", config=vars(args))
    
    # Load CLIP model
    model, tokenizer, processor = load_clip_model(config['model']['image_model_name'], device)
    
    # Load dataset
    dataset = CocoImageDataset(
        images_root_path=config['data']['images_path'],
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer,
        split='train',
        transform=None,
        intra_modal_aug=False
    )
    
    # Handle debug mode
    debug_config = config.get('debug', {})
    if debug_config.get('debug_mode', False):
        debug_limit = debug_config.get('debug_samples', 100)
        if len(dataset.samples) > debug_limit:
            logger.warning(f"DEBUG MODE: Truncating dataset to {debug_limit} samples.")
            dataset.samples = dataset.samples[:debug_limit]
    
    n_samples = len(dataset.samples)
    logger.info(f"Loaded {n_samples:,} captions from training split")
    
    # Build image_id -> caption_indices mapping
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
    logger.info(f"Found {n_images:,} unique images")
    
    # Run mining based on modality
    if args.modality == 'visual':
        # Visual mode: extract image embeddings directly
        result = mine_visual(
            model, processor, dataset, image_ids_ordered, args.top_k, device,
            batch_size=args.batch_size, query_chunk_size=args.query_chunk_size
        )
        
    elif args.modality == 'consensus':
        # Consensus mode: extract text embeddings, then compute consensus
        all_text_embeds = compute_text_embeddings(dataset, tokenizer, model, args.batch_size, device)
        del model
        torch.cuda.empty_cache()
        
        result = mine_consensus(
            all_text_embeds, image_to_indices, image_ids_ordered, args.top_k, device,
            query_chunk_size=args.query_chunk_size
        )
        
    else:  # caption
        # Caption mode: extract text embeddings, direct similarity
        all_text_embeds = compute_text_embeddings(dataset, tokenizer, model, args.batch_size, device)
        del model
        torch.cuda.empty_cache()
        
        result = mine_caption(
            all_text_embeds, args.top_k, device, 
            query_chunk_size=args.query_chunk_size
        )
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    torch.save(result, output_path)
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("MINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Modality: {args.modality}")
    logger.info(f"Output mode: {result['mode']}")
    
    if result['mode'] == 'id_mapping':
        logger.info(f"Total images: {len(result['image_ids']):,}")
    logger.info(f"Indices shape: {result['indices'].shape}")
    logger.info(f"Scores shape: {result['scores'].shape}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Score range: [{result['scores'].min():.4f}, {result['scores'].max():.4f}]")
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 60)
    
    # Log to wandb
    wandb.log({
        "n_samples": n_samples,
        "n_images": n_images,
        "score_min": result['scores'].min().item(),
        "score_max": result['scores'].max().item(),
        "score_mean": result['scores'].mean().item()
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()
