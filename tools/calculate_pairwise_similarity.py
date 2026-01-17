#!/usr/bin/env python3
"""
calculate_pairwise_similarity.py
--------------------------------

Fast Pairwise Similarity Mining using Matrix Multiplication.

This script uses fast matrix multiplication to compute pairwise similarities
in 5-10 minutes instead of 5 days. It processes features in chunks for
GPU memory efficiency.

Modes:
- 'simple': Basic pairwise similarity (image-to-image)
- 'joint_maxmin': Joint MAX and MIN mining using caption-level similarities
  (requires dataset info for image-to-caption mapping)

Key Features:
- Loads all features to GPU for maximum speed (deterministic)
- Processes in configurable-sized chunks using massive matrix multiplication
- WandB logging for progress tracking
- No fallback mode (ensures deterministic results)

Usage:
    # Simple mode (image features only)
    python tools/calculate_pairwise_similarity.py \
        --features_path datasets/coco/features.pt \
        --output_path datasets/coco/pairwise_mining.pt \
        --top_k 50 \
        --batch_size 5000 \
        --mode simple

    # Joint MAX/MIN mode (requires dataset)
    python tools/calculate_pairwise_similarity.py \
        --features_path datasets/coco/caption_features.pt \
        --output_path datasets/coco/pairwise_mining_joint.pt \
        --top_k 50 \
        --query_chunk_size 50 \
        --mode joint_maxmin \
        --config config.yaml
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import wandb
import logging
import sys
import yaml
from collections import defaultdict

# Add parent directory to path for dataset imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_image_to_caption_map(dataset):
    """
    Build a mapping from image_id to list of caption indices.
    
    Args:
        dataset: Dataset with samples containing 'image_id' and 'caption'
    
    Returns:
        image_to_indices: {image_id: [idx0, idx1, idx2, ...]}
        image_ids_ordered: List of unique image_ids in order of first appearance
    """
    logger.info("Building image-to-caption index map...")
    
    image_to_indices = defaultdict(list)
    image_ids_ordered = []
    seen_images = set()
    
    for idx, sample in enumerate(dataset.samples):
        image_id = sample['image_id']
        image_to_indices[image_id].append(idx)
        
        # Track order of first appearance
        if image_id not in seen_images:
            image_ids_ordered.append(image_id)
            seen_images.add(image_id)
    
    # Convert to regular dict
    image_to_indices = dict(image_to_indices)
    
    # Statistics
    n_images = len(image_to_indices)
    captions_per_image = [len(v) for v in image_to_indices.values()]
    
    logger.info(f"  Unique images: {n_images:,}")
    logger.info(f"  Captions per image: min={min(captions_per_image)}, max={max(captions_per_image)}, "
                f"mean={sum(captions_per_image)/len(captions_per_image):.2f}")
    
    return image_to_indices, image_ids_ordered


def mine_joint_max_min(all_embeds, image_to_indices, image_ids, top_k, device, query_chunk_size=50):
    """
    Mines both MAX and MIN pairwise similarities in a single pass.
    
    Structure:
    - Queries: Processed in chunks (e.g., 50 images -> 250 captions).
    - Keys: Full dataset (all captions on GPU).
    
    Efficiency:
    - Replaces nested loops with Block Matrix Multiplication.
    - Calculates [Batch x 5] @ [All x 5].T once.
    - Reshapes to [Batch, 5, All, 5] to extract Max/Min per image pair.
    
    Args:
        all_embeds: [N_total_captions, Dim] - All caption embeddings
        image_to_indices: {image_id: [idx0, idx1, ...]} - Mapping
        image_ids: List of image_ids in order
        top_k: Number of neighbors to find
        device: torch device
        query_chunk_size: Number of query images per chunk
    
    Returns:
        max_indices: [N_images, top_k] - Top-K neighbor indices (MAX mode)
        max_scores: [N_images, top_k] - Scores (MAX mode)
        min_indices: [N_images, top_k] - Top-K neighbor indices (MIN mode)
        min_scores: [N_images, top_k] - Scores (MIN mode)
    """
    n_images = len(image_ids)
    n_caps_total = all_embeds.shape[0]
    dim = all_embeds.shape[1]
    
    # Reshape embeddings to [N_images, 5, Dim] for vectorized operations
    # TRICK: We re-order embeddings into a perfect [N_images, 5, Dim] tensor.
    # If an image has >5 caps, we take first 5. If <5, we repeat.
    # This enables 4D tensor operations which are blazing fast.
    
    logger.info("Reshaping embeddings to [N, 5, Dim] for vectorized operations...")
    structured_embeds = torch.zeros(n_images, 5, dim, device=device)
    
    # Move raw to GPU for fast gather
    all_embeds_gpu = all_embeds.to(device)
    
    for i, img_id in enumerate(tqdm(image_ids, desc="Reshaping embeddings")):
        indices = image_to_indices[img_id]
        # Take first 5, or pad if needed (COCO is usually 5)
        indices = indices[:5]
        if len(indices) < 5:
            indices = indices + [indices[0]] * (5 - len(indices))
            
        structured_embeds[i] = all_embeds_gpu[indices]
    
    del all_embeds_gpu
    torch.cuda.empty_cache()
    
    # Result containers
    max_indices_list = []
    max_scores_list = []
    min_indices_list = []
    min_scores_list = []
    
    logger.info(f"Mining Joint Max/Min... (Query Chunk: {query_chunk_size})")
    
    # Keys: All images [N, 5, Dim]
    keys = structured_embeds
    # Flatten Keys for dot product: [N*5, Dim]
    keys_flat = keys.view(-1, dim)
    
    n_chunks = (n_images + query_chunk_size - 1) // query_chunk_size
    
    for i in tqdm(range(0, n_images, query_chunk_size), desc="Joint Mining"):
        end = min(i + query_chunk_size, n_images)
        batch_size = end - i
        
        # Query: [B, 5, Dim]
        queries = structured_embeds[i:end]
        queries_flat = queries.view(-1, dim)  # [B*5, Dim]
        
        # --- THE HEAVY LIFTING (ONCE) ---
        # Sim: [B*5, N*5]
        # This calculates similarity of every query caption against every key caption
        full_sim = queries_flat @ keys_flat.T
        
        # Reshape to isolate Image-to-Image blocks
        # Target: [B, 5, N, 5]
        # B: Query Images, N: Key Images, 5: Query Caps, 5: Key Caps
        # Current: [B*5, N*5] -> View as [B, 5, N, 5]
        full_sim = full_sim.view(batch_size, 5, n_images, 5)
        
        # --- EXTRACT MAX ---
        # "Optimistic": Max similarity among the 25 pairs
        # Collapse the 5x5 dimensions -> [B, N]
        max_sim_img, _ = full_sim.max(dim=3)  # Max over key caps -> [B, 5, N]
        max_sim_img, _ = max_sim_img.max(dim=1)  # Max over query caps -> [B, N]
        
        # Mask Self
        for b in range(batch_size):
            max_sim_img[b, i + b] = float('-inf')
            
        # Top K Max
        batch_max_scores, batch_max_indices = max_sim_img.topk(top_k, dim=1)
        max_indices_list.append(batch_max_indices.cpu())
        max_scores_list.append(batch_max_scores.cpu())
        
        # --- EXTRACT MIN ---
        # "Pessimistic": Min similarity among the 25 pairs
        # Collapse 5x5 dimensions -> [B, N]
        min_sim_img, _ = full_sim.min(dim=3)  # Min over key caps
        min_sim_img, _ = min_sim_img.min(dim=1)  # Min over query caps
        
        # Mask Self (We want Min, so self (1.0) is not a threat for top-k,
        # but technically we shouldn't retrieve self as a neighbor)
        for b in range(batch_size):
            min_sim_img[b, i + b] = float('inf')
        
        # Top K Min (we want lowest scores, so we take smallest values)
        # But topk gives largest, so we negate, take topk, then negate back
        batch_min_scores, batch_min_indices = (-min_sim_img).topk(top_k, dim=1)
        batch_min_scores = -batch_min_scores  # Negate back
        
        min_indices_list.append(batch_min_indices.cpu())
        min_scores_list.append(batch_min_scores.cpu())
        
        del full_sim, queries_flat
        torch.cuda.empty_cache()
        
        # Log progress to WandB every 10 chunks or at the end
        chunk_idx = i // query_chunk_size
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            progress = (chunk_idx + 1) / n_chunks * 100
            wandb.log({
                "mining_progress": progress,
                "mining_chunk": chunk_idx + 1,
                "mining_images_processed": end
            })
    
    # Aggregate
    max_indices = torch.cat(max_indices_list, dim=0)  # [N, top_k]
    max_scores = torch.cat(max_scores_list, dim=0)    # [N, top_k]
    min_indices = torch.cat(min_indices_list, dim=0)  # [N, top_k]
    min_scores = torch.cat(min_scores_list, dim=0)    # [N, top_k]
    
    return max_indices, max_scores, min_indices, min_scores


def mine_simple(gpu_features, num_images, top_k, batch_size, device):
    """
    Simple pairwise similarity mining (image-to-image).
    
    Args:
        gpu_features: [N, Dim] - All image features on GPU
        num_images: Number of images
        top_k: Number of neighbors to find
        batch_size: Chunk size for processing
        device: torch device
    
    Returns:
        final_indices: [N, K+1] - Top-K neighbor indices
        final_scores: [N, K+1] - Scores
    """
    all_top_indices = []
    all_top_scores = []
    
    n_chunks = (num_images + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, num_images, batch_size), desc="Mining Hard Negatives"):
        end = min(i + batch_size, num_images)
        
        # Query Chunk
        query_chunk = gpu_features[i:end]
        
        # Sim Matrix: (Batch_Size, N)
        sim_matrix = torch.matmul(query_chunk, gpu_features.T)
        
        # Mask self-similarity (diagonal within chunk)
        chunk_size = end - i
        for b in range(chunk_size):
            sim_matrix[b, i + b] = float('-inf')
        
        # Find Top K+1 values (Largest scores)
        scores, indices = torch.topk(sim_matrix, k=top_k + 1, dim=1)
        
        all_top_indices.append(indices.cpu())
        all_top_scores.append(scores.cpu())
        
        del sim_matrix, query_chunk
        torch.cuda.empty_cache()
        
        # Log progress to WandB every 10 chunks or at the end
        chunk_idx = i // batch_size
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            progress = (chunk_idx + 1) / n_chunks * 100
            wandb.log({
                "mining_progress": progress,
                "mining_chunk": chunk_idx + 1,
                "mining_images_processed": end
            })
    
    # Aggregate
    final_indices = torch.cat(all_top_indices, dim=0)  # (N, K+1)
    final_scores = torch.cat(all_top_scores, dim=0)    # (N, K+1)
    
    return final_indices, final_scores


def main():
    parser = argparse.ArgumentParser(
        description="Fast Pairwise Similarity Mining using Matrix Multiplication"
    )
    parser.add_argument(
        '--features_path', type=str, required=True,
        help='Path to saved .pt features (N, D) or (N_captions, D) for joint_maxmin mode'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Where to save the mining results'
    )
    parser.add_argument(
        '--top_k', type=int, default=50,
        help='Number of hard negatives/similars to save per image'
    )
    parser.add_argument(
        '--batch_size', type=int, default=5000,
        help='Chunk size for GPU memory safety (simple mode)'
    )
    parser.add_argument(
        '--query_chunk_size', type=int, default=50,
        help='Query chunk size for joint_maxmin mode'
    )
    parser.add_argument(
        '--mode', type=str, default='simple', choices=['simple', 'joint_maxmin'],
        help='Mining mode: simple (image-to-image) or joint_maxmin (caption-level)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Config file path (required for joint_maxmin mode)'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='pairwise-mining',
        help='WandB project name'
    )
    parser.add_argument(
        '--wandb_name', type=str, default=None,
        help='WandB run name (default: auto-generated)'
    )
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="pairwise_mining",
        config={
            "features_path": args.features_path,
            "output_path": args.output_path,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "mode": args.mode
        }
    )

    print(f"🔄 Loading features from {args.features_path}...")
    # Load features (start on CPU)
    features = torch.load(args.features_path, map_location='cpu')
    
    # If dict, get 'features' key, otherwise use tensor directly
    if isinstance(features, dict):
        if 'features' in features:
            features = features['features']
        else:
            print("⚠️ Warning: Dictionary loaded but 'features' key not found. Using the first value.")
            features = list(features.values())[0]

    num_samples = features.size(0)
    feature_dim = features.size(1)
    print(f"✅ Loaded {num_samples} samples. Feature dim: {feature_dim}")

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Please use a GPU-enabled environment.")
    
    device = torch.device('cuda')
    print(f"🚀 Running on: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Log to WandB
    wandb.log({
        "num_samples": num_samples,
        "feature_dim": feature_dim,
        "gpu_name": torch.cuda.get_device_name(0),
        "mode": args.mode
    })
    
    # Normalize features (required for Cosine Similarity)
    print("📐 Normalizing features for cosine similarity...")
    features = F.normalize(features.float(), p=2, dim=1)

    # Move all features to GPU (deterministic - no fallback)
    print("📦 Moving all features to GPU...")
    try:
        gpu_features = features.to(device)
        print("✅ All features moved to GPU for maximum speed.")
        
        # Log memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        print(f"   GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
        wandb.log({
            "gpu_memory_allocated_gb": memory_allocated,
            "gpu_memory_reserved_gb": memory_reserved
        })
    except RuntimeError as e:
        logger.error(f"Failed to move features to GPU: {e}")
        logger.error("This script requires all features to fit in GPU memory for deterministic results.")
        raise

    # Run mining based on mode
    if args.mode == 'simple':
        print("\n🔍 Running SIMPLE mode (image-to-image)...")
        final_indices, final_scores = mine_simple(
            gpu_features, num_samples, args.top_k, args.batch_size, device
        )
        
        save_dict = {
            "indices": final_indices,
            "scores": final_scores,
            "metadata": {
                "top_k": args.top_k,
                "num_images": num_samples,
                "feature_dim": feature_dim,
                "batch_size": args.batch_size,
                "mode": "simple"
            }
        }
        
    elif args.mode == 'joint_maxmin':
        print("\n🔍 Running JOINT MAX/MIN mode (caption-level)...")
        
        if args.config is None:
            raise ValueError("--config is required for joint_maxmin mode")
        
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load dataset to build image-to-caption mapping
        from src.data import CocoImageDataset
        from transformers import CLIPTokenizer
        
        tokenizer = CLIPTokenizer.from_pretrained(config['model']['image_model_name'])
        max_length = config['data'].get('max_length', 77)
        
        dataset = CocoImageDataset(
            images_root_path=config['data']['images_path'],
            captions_path=config['data']['captions_path'],
            tokenizer=tokenizer,
            max_length=max_length,
            split='train',
            transform=None,
            intra_modal_aug=False
        )
        
        # Build mapping
        image_to_indices, image_ids_ordered = build_image_to_caption_map(dataset)
        n_images = len(image_ids_ordered)
        
        # Verify feature count matches caption count
        if num_samples != len(dataset.samples):
            raise ValueError(
                f"Feature count ({num_samples}) doesn't match dataset caption count ({len(dataset.samples)})"
            )
        
        wandb.log({
            "num_images": n_images,
            "num_captions": num_samples
        })
        
        # Run joint mining
        max_indices, max_scores, min_indices, min_scores = mine_joint_max_min(
            gpu_features, image_to_indices, image_ids_ordered,
            args.top_k, device, args.query_chunk_size
        )
        
        save_dict = {
            "max_indices": max_indices,
            "max_scores": max_scores,
            "min_indices": min_indices,
            "min_scores": min_scores,
            "metadata": {
                "top_k": args.top_k,
                "num_images": n_images,
                "num_captions": num_samples,
                "feature_dim": feature_dim,
                "query_chunk_size": args.query_chunk_size,
                "mode": "joint_maxmin"
            }
        }
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Saving to {args.output_path}...")
    torch.save(save_dict, args.output_path)
    
    # Get file size
    file_size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    
    # Print statistics
    print("\n✅ Mining complete!")
    if args.mode == 'simple':
        print(f"   Total images: {num_samples:,}")
        print(f"   Top-K per image: {args.top_k}")
        print(f"   Output shape: indices={final_indices.shape}, scores={final_scores.shape}")
        print(f"   Score range: [{final_scores.min():.4f}, {final_scores.max():.4f}]")
        print(f"   Mean score: {final_scores.mean():.4f}")
        
        # Log final metrics to WandB
        wandb.log({
            "output_file_size_mb": file_size_mb,
            "score_min": final_scores.min().item(),
            "score_max": final_scores.max().item(),
            "score_mean": final_scores.mean().item(),
            "score_std": final_scores.std().item()
        })
    else:  # joint_maxmin
        print(f"   Total images: {n_images:,}")
        print(f"   Total captions: {num_samples:,}")
        print(f"   Top-K per image: {args.top_k}")
        print(f"   MAX - Output shape: indices={max_indices.shape}, scores={max_scores.shape}")
        print(f"   MAX - Score range: [{max_scores.min():.4f}, {max_scores.max():.4f}]")
        print(f"   MAX - Mean score: {max_scores.mean():.4f}")
        print(f"   MIN - Output shape: indices={min_indices.shape}, scores={min_scores.shape}")
        print(f"   MIN - Score range: [{min_scores.min():.4f}, {min_scores.max():.4f}]")
        print(f"   MIN - Mean score: {min_scores.mean():.4f}")
        
        # Log final metrics to WandB
        wandb.log({
            "output_file_size_mb": file_size_mb,
            "max_score_min": max_scores.min().item(),
            "max_score_max": max_scores.max().item(),
            "max_score_mean": max_scores.mean().item(),
            "min_score_min": min_scores.min().item(),
            "min_score_max": min_scores.max().item(),
            "min_score_mean": min_scores.mean().item()
        })
    
    print(f"   File size: {file_size_mb:.2f} MB")
    
    wandb.finish()


if __name__ == "__main__":
    main()
