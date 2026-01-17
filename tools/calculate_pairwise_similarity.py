#!/usr/bin/env python3
"""
calculate_pairwise_similarity.py
--------------------------------
End-to-End Joint Pairwise Similarity Mining.

Flow:
1. Load CLIP Model & Dataset.
2. Extract all Caption Embeddings on-the-fly (RAM only).
3. DELETE CLIP Model to free VRAM.
4. Perform Block Matrix Multiplication (Joint Max/Min).
5. Save only the final indices/scores.

Usage:
    python tools/calculate_pairwise_similarity.py \
        --config config.yaml \
        --output_path datasets/coco/pairwise_mining_joint.pt \
        --top_k 50 \
        --query_chunk_size 100
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
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return model, tokenizer


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
            all_embeds.append(embeds.cpu())  # Keep on CPU until needed
    
    return torch.cat(all_embeds, dim=0)


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
    dim = all_embeds.shape[1]
    
    # Reshape embeddings to [N_images, 5, Dim] for vectorized operations
    # TRICK: We re-order embeddings into a perfect [N_images, 5, Dim] tensor.
    # If an image has >5 caps, we take first 5. If <5, we repeat.
    # This enables 4D tensor operations which are blazing fast.
    
    logger.info("Reshaping embeddings to [N, 5, Dim] for vectorized mining...")
    structured_embeds = torch.zeros(n_images, 5, dim, device=device)
    
    # Move raw to GPU for fast gather
    all_embeds_gpu = all_embeds.to(device)
    
    for i, img_id in enumerate(tqdm(image_ids, desc="Reshaping")):
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


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Joint Pairwise Similarity Mining"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Where to save the mining results'
    )
    parser.add_argument(
        '--top_k', type=int, default=50,
        help='Number of neighbors to find per image'
    )
    parser.add_argument(
        '--query_chunk_size', type=int, default=100,
        help='Query chunk size for joint_maxmin mode'
    )
    parser.add_argument(
        '--extraction_batch_size', type=int, default=512,
        help='Batch size for feature extraction'
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

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Please use a GPU-enabled environment.")
    
    device = torch.device('cuda')
    logger.info(f"🚀 Running on: {device}")
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config
    clip_model_name = config['model']['image_model_name']
    max_length = config['data'].get('max_length', 77)
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="pairwise_mining_joint",
        config={
            "config_path": args.config,
            "output_path": args.output_path,
            "top_k": args.top_k,
            "query_chunk_size": args.query_chunk_size,
            "extraction_batch_size": args.extraction_batch_size,
            "clip_model": clip_model_name
        }
    )
    
    wandb.log({
        "gpu_name": torch.cuda.get_device_name(0)
    })

    # =========================================================
    # Phase 1: Load CLIP Model & Dataset
    # =========================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Loading CLIP Model & Dataset")
    logger.info("=" * 60)
    
    clip_model, tokenizer = load_clip_model(clip_model_name, device)
    
    dataset = CocoImageDataset(
        images_root_path=config['data']['images_path'],
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer,
        max_length=max_length,
        split='train',
        transform=None,
        intra_modal_aug=False
    )
    
    # Respect debug mode
    debug_config = config.get('debug', {})
    if debug_config.get('debug_mode', False):
        debug_limit = debug_config.get('debug_samples', 100)
        if len(dataset.samples) > debug_limit:
            logger.warning(f"DEBUG MODE: Truncating to {debug_limit} samples")
            dataset.samples = dataset.samples[:debug_limit]
    
    n_captions = len(dataset.samples)
    logger.info(f"Total captions: {n_captions:,}")
    
    # Build image-to-caption mapping
    image_to_indices, image_ids_ordered = build_image_to_caption_map(dataset)
    n_images = len(image_ids_ordered)
    
    wandb.log({
        "total_captions": n_captions,
        "total_images": n_images
    })

    # =========================================================
    # Phase 2: Extract All Caption Embeddings
    # =========================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Extracting Caption Embeddings")
    logger.info("=" * 60)
    
    all_embeddings = compute_all_caption_embeddings(
        dataset=dataset,
        tokenizer=tokenizer,
        model=clip_model,
        batch_size=args.extraction_batch_size,
        device=device,
        max_len=max_length
    )
    
    logger.info(f"✅ Extracted embeddings shape: {all_embeddings.shape}")
    
    # =========================================================
    # Phase 3: Free CLIP Model (Free VRAM for Mining)
    # =========================================================
    logger.info("=" * 60)
    logger.info("Phase 3: Freeing CLIP Model")
    logger.info("=" * 60)
    
    del clip_model
    del tokenizer
    torch.cuda.empty_cache()
    
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    logger.info(f"GPU Memory after cleanup: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    wandb.log({
        "gpu_memory_after_cleanup_gb": memory_allocated
    })

    # =========================================================
    # Phase 4: Joint MAX/MIN Mining
    # =========================================================
    logger.info("=" * 60)
    logger.info("Phase 4: Joint MAX/MIN Mining")
    logger.info("=" * 60)
    
    max_indices, max_scores, min_indices, min_scores = mine_joint_max_min(
        all_embeddings=all_embeddings,
        image_to_indices=image_to_indices,
        image_ids=image_ids_ordered,
        top_k=args.top_k,
        device=device,
        query_chunk_size=args.query_chunk_size
    )
    
    # Free embeddings
    del all_embeddings
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 5: Save Results
    # =========================================================
    logger.info("=" * 60)
    logger.info("Phase 5: Saving Results")
    logger.info("=" * 60)
    
    save_dict = {
        "max_indices": max_indices,
        "max_scores": max_scores,
        "min_indices": min_indices,
        "min_scores": min_scores,
        "metadata": {
            "top_k": args.top_k,
            "num_images": n_images,
            "num_captions": n_captions,
            "query_chunk_size": args.query_chunk_size,
            "mode": "joint_maxmin"
        }
    }
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"💾 Saving to {args.output_path}...")
    torch.save(save_dict, args.output_path)
    
    # Get file size
    file_size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("MINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total images: {n_images:,}")
    logger.info(f"Total captions: {n_captions:,}")
    logger.info(f"Top-K per image: {args.top_k}")
    logger.info(f"\nMAX Mode:")
    logger.info(f"  Output shape: indices={max_indices.shape}, scores={max_scores.shape}")
    logger.info(f"  Score range: [{max_scores.min():.4f}, {max_scores.max():.4f}]")
    logger.info(f"  Mean score: {max_scores.mean():.4f}")
    logger.info(f"\nMIN Mode:")
    logger.info(f"  Output shape: indices={min_indices.shape}, scores={min_scores.shape}")
    logger.info(f"  Score range: [{min_scores.min():.4f}, {min_scores.max():.4f}]")
    logger.info(f"  Mean score: {min_scores.mean():.4f}")
    logger.info(f"\nFile size: {file_size_mb:.2f} MB")
    logger.info("=" * 60)
    
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
    
    wandb.finish()
    logger.info("\n✅ Done!")


if __name__ == "__main__":
    main()
