#!/usr/bin/env python3
"""
calculate_pairwise_similarity.py
--------------------------------

Offline Pairwise Image Similarity Mining for Knowledge Distillation.

This script computes image-to-image similarity based on the semantic consensus
of their captions, using an **Index Mapping** strategy to preserve original
dataset indices for DistillationLoss compatibility.

**Key Design: Index Mapping (Not Physical Grouping)**
    - Embeddings are kept FLAT: [N_total_captions, Dim]
    - An index map `image_to_caption_indices = {image_id: [idx0, idx1, ...]}` is built
    - Mining uses the map to retrieve caption embeddings for each image
    - This preserves original indices and handles variable captions per image

Aggregation Modes:
- 'max':  Score = max value in MxN matrix (Optimistic: best caption pair)
- 'mean': Score = mean(query_caps) @ mean(key_caps) (Semantic consensus) [RECOMMENDED]
- 'min':  Score = min value in MxN matrix (Pessimistic: worst caption pair)

Output Format:
- indices: [N_total_captions, top_k] - Each row contains top-k neighbor *caption* indices
- scores:  [N_total_captions, top_k] - Corresponding similarity scores

Mapping Strategy (Image → Caption):
    If Image B (with first caption index `j`) is a neighbor of Image A:
    - For EVERY original caption index `i` belonging to Image A:
        - Assign `j` (representative caption of Image B) as the target

Usage:
    python tools/calculate_pairwise_similarity.py --config config.yaml --mode mean
    python tools/calculate_pairwise_similarity.py --config config.yaml --mode max --batch_size 1024
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
from pathlib import Path
from transformers import CLIPModel, CLIPTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CocoImageDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_clip_text_encoder(model_name: str, device: torch.device):
    """Load CLIP model for text encoding only."""
    logger.info(f"Loading CLIP model: {model_name}")
    
    clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    for param in clip_model.parameters():
        param.requires_grad = False
    
    logger.info(f"CLIP projection dim: {clip_model.config.projection_dim}")
    
    return clip_model, tokenizer


def build_image_to_caption_map(dataset: CocoImageDataset) -> dict:
    """
    Build a mapping from image_id to list of caption indices.
    
    Args:
        dataset: CocoImageDataset instance
    
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


def compute_all_text_embeddings(
    clip_model: CLIPModel,
    dataset: CocoImageDataset,
    tokenizer: CLIPTokenizer,
    batch_size: int,
    device: torch.device,
    max_length: int
) -> torch.Tensor:
    """
    Compute normalized text embeddings for all captions.
    Returns FLAT embeddings on GPU in FP16: [N_total_captions, Dim]
    """
    n_samples = len(dataset.samples)
    embed_dim = clip_model.config.projection_dim
    
    logger.info(f"Computing text embeddings for {n_samples:,} captions...")
    logger.info(f"Batch size: {batch_size}, Embed dim: {embed_dim}")
    
    all_embeddings = torch.zeros(n_samples, embed_dim, dtype=torch.float16, device=device)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            captions = [dataset.samples[i]['caption'] for i in range(start_idx, end_idx)]
            
            tokenized = tokenizer(
                captions,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                text_embeds = clip_model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
            all_embeddings[start_idx:end_idx] = text_embeds.half()
            
            if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                progress = (batch_idx + 1) / n_batches * 100
                wandb.log({
                    "extraction_progress": progress,
                    "extraction_samples_processed": end_idx
                })
                logger.info(f"Extraction progress: {progress:.1f}% ({end_idx:,}/{n_samples:,})")
    
    logger.info(f"Embeddings shape (FLAT): {all_embeddings.shape}")
    return all_embeddings


def compute_image_mean_embeddings(
    all_embeddings: torch.Tensor,
    image_to_indices: dict,
    image_ids_ordered: list,
    device: torch.device
) -> tuple:
    """
    Compute mean embedding for each image from its caption embeddings.
    
    Args:
        all_embeddings: [N_total_captions, Dim] - FLAT embeddings
        image_to_indices: {image_id: [idx0, idx1, ...]}
        image_ids_ordered: List of image_ids in order
        device: torch device
    
    Returns:
        mean_embeddings: [N_images, Dim] - Mean embedding per image
        image_id_to_img_idx: {image_id: img_idx} - Mapping to mean_embeddings index
    """
    n_images = len(image_ids_ordered)
    embed_dim = all_embeddings.shape[1]
    
    logger.info(f"Computing mean embeddings for {n_images:,} images...")
    
    mean_embeddings = torch.zeros(n_images, embed_dim, dtype=torch.float32, device=device)
    image_id_to_img_idx = {}
    
    for img_idx, image_id in enumerate(image_ids_ordered):
        caption_indices = image_to_indices[image_id]
        
        # Get caption embeddings for this image
        cap_embeds = all_embeddings[caption_indices].float()  # [n_caps, Dim]
        
        # Mean and normalize
        mean_embed = cap_embeds.mean(dim=0)
        mean_embed = F.normalize(mean_embed, p=2, dim=0)
        
        mean_embeddings[img_idx] = mean_embed
        image_id_to_img_idx[image_id] = img_idx
    
    return mean_embeddings.half(), image_id_to_img_idx


def mine_pairwise_similarity_mean(
    mean_embeddings: torch.Tensor,
    top_k: int,
    query_chunk_size: int = 500,
    device: torch.device = None
) -> tuple:
    """
    Compute pairwise image similarity using MEAN aggregation.
    
    Args:
        mean_embeddings: [N_images, Dim] - Pre-computed mean embeddings
        top_k: Number of neighbors to find
        query_chunk_size: Chunk size for memory efficiency
        device: torch device
    
    Returns:
        indices: [N_images, top_k] - Top-K neighbor image indices (in mean_embeddings order)
        scores: [N_images, top_k] - Similarity scores
    """
    n_images = mean_embeddings.shape[0]
    
    logger.info(f"Mining pairwise similarity (mode=mean) for {n_images:,} images...")
    
    all_indices = []
    all_scores = []
    
    n_chunks = (n_images + query_chunk_size - 1) // query_chunk_size
    
    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * query_chunk_size
            end_idx = min(start_idx + query_chunk_size, n_images)
            
            query_chunk = mean_embeddings[start_idx:end_idx]  # [chunk_size, Dim]
            
            # Similarity: [chunk_size, N_images]
            sims = query_chunk.float() @ mean_embeddings.float().T
            
            # Mask self-similarity
            chunk_size = end_idx - start_idx
            for i in range(chunk_size):
                sims[i, start_idx + i] = float('-inf')
            
            # Top-K
            top_k_scores, top_k_indices = sims.topk(k=top_k, dim=1)
            
            all_indices.append(top_k_indices.cpu())
            all_scores.append(top_k_scores.cpu())
            
            if (chunk_idx + 1) % 20 == 0 or chunk_idx == n_chunks - 1:
                progress = (chunk_idx + 1) / n_chunks * 100
                wandb.log({"mining_progress": progress})
                logger.info(f"Mining progress: {progress:.1f}%")
    
    indices = torch.cat(all_indices, dim=0)
    scores = torch.cat(all_scores, dim=0)
    
    return indices, scores


def mine_pairwise_similarity_maxmin(
    all_embeddings: torch.Tensor,
    image_to_indices: dict,
    image_ids_ordered: list,
    top_k: int,
    mode: str,  # 'max' or 'min'
    query_chunk_size: int = 100,
    key_chunk_size: int = 500,
    device: torch.device = None
) -> tuple:
    """
    Compute pairwise image similarity using MAX or MIN aggregation.
    
    For each image pair, compute the full MxN similarity matrix between their
    caption embeddings and take max/min.
    
    Args:
        all_embeddings: [N_total_captions, Dim] - FLAT embeddings
        image_to_indices: {image_id: [idx0, idx1, ...]}
        image_ids_ordered: List of image_ids in order
        top_k: Number of neighbors to find
        mode: 'max' or 'min'
        query_chunk_size: Number of query images per chunk
        key_chunk_size: Number of key images per chunk
        device: torch device
    
    Returns:
        indices: [N_images, top_k] - Top-K neighbor image indices
        scores: [N_images, top_k] - Similarity scores
    """
    n_images = len(image_ids_ordered)
    
    logger.info(f"Mining pairwise similarity (mode={mode}) for {n_images:,} images...")
    logger.info(f"This requires MxN matrix per pair - may be slower than mean mode...")
    
    all_indices = []
    all_scores = []
    
    n_query_chunks = (n_images + query_chunk_size - 1) // query_chunk_size
    
    with torch.no_grad():
        for q_chunk_idx in range(n_query_chunks):
            q_start = q_chunk_idx * query_chunk_size
            q_end = min(q_start + query_chunk_size, n_images)
            q_size = q_end - q_start
            
            # Initialize running top-k for this query chunk
            running_scores = torch.full((q_size, top_k + 1), float('-inf'), device=device)
            running_indices = torch.zeros((q_size, top_k + 1), dtype=torch.long, device=device)
            
            n_key_chunks = (n_images + key_chunk_size - 1) // key_chunk_size
            
            for k_chunk_idx in range(n_key_chunks):
                k_start = k_chunk_idx * key_chunk_size
                k_end = min(k_start + key_chunk_size, n_images)
                k_size = k_end - k_start
                
                # Compute chunk similarity matrix [q_size, k_size]
                chunk_scores = torch.zeros(q_size, k_size, device=device)
                
                for qi, q_img_idx in enumerate(range(q_start, q_end)):
                    q_image_id = image_ids_ordered[q_img_idx]
                    q_cap_indices = image_to_indices[q_image_id]
                    q_embeds = all_embeddings[q_cap_indices].float()  # [n_q_caps, Dim]
                    
                    for ki, k_img_idx in enumerate(range(k_start, k_end)):
                        k_image_id = image_ids_ordered[k_img_idx]
                        k_cap_indices = image_to_indices[k_image_id]
                        k_embeds = all_embeddings[k_cap_indices].float()  # [n_k_caps, Dim]
                        
                        # Compute MxN similarity matrix
                        sim_matrix = q_embeds @ k_embeds.T  # [n_q_caps, n_k_caps]
                        
                        # Aggregate
                        if mode == 'max':
                            chunk_scores[qi, ki] = sim_matrix.max()
                        else:  # min
                            chunk_scores[qi, ki] = sim_matrix.min()
                
                # Mask self-similarity
                for qi in range(q_size):
                    global_q_idx = q_start + qi
                    if k_start <= global_q_idx < k_end:
                        local_k_idx = global_q_idx - k_start
                        chunk_scores[qi, local_k_idx] = float('-inf')
                
                # Get top-k from this chunk
                chunk_k = min(top_k + 1, k_size)
                chunk_top_scores, chunk_top_indices = chunk_scores.topk(k=chunk_k, dim=1)
                chunk_top_indices = chunk_top_indices + k_start  # Global image indices
                
                # Merge with running top-k
                merged_scores = torch.cat([running_scores, chunk_top_scores], dim=1)
                merged_indices = torch.cat([running_indices, chunk_top_indices], dim=1)
                
                _, topk_positions = merged_scores.topk(k=top_k + 1, dim=1)
                running_scores = torch.gather(merged_scores, 1, topk_positions)
                running_indices = torch.gather(merged_indices, 1, topk_positions)
            
            # Take top_k
            all_indices.append(running_indices[:, :top_k].cpu())
            all_scores.append(running_scores[:, :top_k].cpu())
            
            progress = (q_chunk_idx + 1) / n_query_chunks * 100
            wandb.log({"mining_progress": progress})
            logger.info(f"Mining progress: {progress:.1f}%")
    
    indices = torch.cat(all_indices, dim=0)
    scores = torch.cat(all_scores, dim=0)
    
    return indices, scores


def map_image_results_to_caption_indices(
    image_neighbor_indices: torch.Tensor,
    image_neighbor_scores: torch.Tensor,
    image_to_indices: dict,
    image_ids_ordered: list,
    n_total_captions: int
) -> tuple:
    """
    Map image-level mining results to caption indices.
    
    For each image A with neighbor image B (score S):
    - Get representative caption index of B: first index in image_to_indices[B]
    - For EVERY caption index belonging to A: assign (representative_idx, S) as target
    
    Args:
        image_neighbor_indices: [N_images, top_k] - Neighbor image indices
        image_neighbor_scores: [N_images, top_k] - Scores
        image_to_indices: {image_id: [idx0, idx1, ...]}
        image_ids_ordered: List of image_ids in order
        n_total_captions: Total number of captions
    
    Returns:
        caption_indices: [N_total_captions, top_k] - Target caption indices
        caption_scores: [N_total_captions, top_k] - Scores
    """
    n_images, top_k = image_neighbor_indices.shape
    
    logger.info(f"Mapping image results to caption indices...")
    logger.info(f"  Images: {n_images:,}, Captions: {n_total_captions:,}, Top-K: {top_k}")
    
    # Pre-compute representative caption index for each image
    # Representative = first caption index
    img_idx_to_representative_cap = {}
    for img_idx, image_id in enumerate(image_ids_ordered):
        cap_indices = image_to_indices[image_id]
        img_idx_to_representative_cap[img_idx] = cap_indices[0]  # First caption
    
    # Initialize output tensors
    caption_indices = torch.zeros(n_total_captions, top_k, dtype=torch.long)
    caption_scores = torch.zeros(n_total_captions, top_k, dtype=torch.float32)
    
    # Fill in for each image
    for img_idx, image_id in enumerate(image_ids_ordered):
        # Get all caption indices for this query image
        query_cap_indices = image_to_indices[image_id]
        
        # Get neighbor image indices and scores
        neighbor_img_indices = image_neighbor_indices[img_idx]  # [top_k]
        neighbor_scores = image_neighbor_scores[img_idx]  # [top_k]
        
        # Convert neighbor image indices to representative caption indices
        neighbor_cap_indices = torch.tensor([
            img_idx_to_representative_cap[neighbor_img_idx.item()]
            for neighbor_img_idx in neighbor_img_indices
        ], dtype=torch.long)
        
        # Assign to ALL captions of the query image
        for cap_idx in query_cap_indices:
            caption_indices[cap_idx] = neighbor_cap_indices
            caption_scores[cap_idx] = neighbor_scores
    
    logger.info(f"  Output shape: indices={caption_indices.shape}, scores={caption_scores.shape}")
    
    return caption_indices, caption_scores


def main():
    parser = argparse.ArgumentParser(
        description="Offline Pairwise Image Similarity Mining for Knowledge Distillation"
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=['max', 'mean', 'min'],
        help='Aggregation mode: max (optimistic), mean (consensus), min (pessimistic)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=512,
        help='Batch size for embedding computation'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path (default: datasets/coco/pairwise_mining_{mode}.pt)'
    )
    
    args = parser.parse_args()
    
    # Device setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficient mining.")
    
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config
    distillation_config = config.get('distillation', {})
    top_k = distillation_config.get('top_k', 30)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"datasets/coco/pairwise_mining_{args.mode}.pt"
    
    logger.info(f"Mode: {args.mode}, Top-K: {top_k}")
    logger.info(f"Output: {output_path}")
    
    # Get CLIP model name
    clip_model_name = config['model']['image_model_name']
    max_length = config['data'].get('max_length', 77)
    
    # Initialize wandb
    logging_config = config.get('logging', {})
    wandb_project = logging_config.get('wandb_project', 'coco-distillation')
    
    wandb.init(
        project=wandb_project,
        job_type="pairwise_mining",
        config={
            "mode": args.mode,
            "batch_size": args.batch_size,
            "top_k": top_k,
            "output_path": output_path,
            "clip_model": clip_model_name
        }
    )
    
    # Load CLIP model and tokenizer
    clip_model, tokenizer = load_clip_text_encoder(clip_model_name, device)
    
    # Load dataset
    logger.info("Loading COCO training dataset...")
    
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
    
    # =========================================================
    # Phase 1: Build Image-to-Caption Index Map
    # =========================================================
    image_to_indices, image_ids_ordered = build_image_to_caption_map(dataset)
    n_images = len(image_ids_ordered)
    
    wandb.log({"total_captions": n_captions, "total_images": n_images})
    
    # =========================================================
    # Phase 2: Compute Text Embeddings (FLAT - preserve indices)
    # =========================================================
    all_embeddings = compute_all_text_embeddings(
        clip_model=clip_model,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        max_length=max_length
    )
    
    # Free CLIP model
    del clip_model
    torch.cuda.empty_cache()
    
    # =========================================================
    # Phase 3: Mine Pairwise Similarity (using Index Map)
    # =========================================================
    if args.mode == 'mean':
        # Optimized path: compute mean embeddings first
        mean_embeddings, _ = compute_image_mean_embeddings(
            all_embeddings=all_embeddings,
            image_to_indices=image_to_indices,
            image_ids_ordered=image_ids_ordered,
            device=device
        )
        
        image_indices, image_scores = mine_pairwise_similarity_mean(
            mean_embeddings=mean_embeddings,
            top_k=top_k,
            query_chunk_size=500,
            device=device
        )
        
        del mean_embeddings
    else:  # max or min
        image_indices, image_scores = mine_pairwise_similarity_maxmin(
            all_embeddings=all_embeddings,
            image_to_indices=image_to_indices,
            image_ids_ordered=image_ids_ordered,
            top_k=top_k,
            mode=args.mode,
            query_chunk_size=50,  # Smaller for max/min due to MxN computation
            key_chunk_size=200,
            device=device
        )
    
    # Free embeddings
    del all_embeddings
    torch.cuda.empty_cache()
    
    # =========================================================
    # Phase 4: Map Image Results to Caption Indices
    # =========================================================
    caption_indices, caption_scores = map_image_results_to_caption_indices(
        image_neighbor_indices=image_indices,
        image_neighbor_scores=image_scores,
        image_to_indices=image_to_indices,
        image_ids_ordered=image_ids_ordered,
        n_total_captions=n_captions
    )
    
    # =========================================================
    # Save Output
    # =========================================================
    output_data = {
        'indices': caption_indices,  # [N_captions, top_k]
        'scores': caption_scores,    # [N_captions, top_k]
        'metadata': {
            'mode': args.mode,
            'top_k': top_k,
            'n_images': n_images,
            'n_captions': n_captions,
            'mapping_strategy': 'index_map'
        }
    }
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    logger.info(f"Saving to {output_path}")
    torch.save(output_data, output_path)
    
    # Log final metrics
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    wandb.log({
        "output_file_size_mb": file_size_mb,
        "score_min": caption_scores.min().item(),
        "score_max": caption_scores.max().item(),
        "score_mean": caption_scores.mean().item()
    })
    
    # =========================================================
    # Summary
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("PAIRWISE MINING COMPLETE (Index Mapping Strategy)")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Total images: {n_images:,}")
    logger.info(f"Total captions: {n_captions:,}")
    logger.info(f"Top-K neighbors: {top_k}")
    logger.info(f"Output shape: indices={caption_indices.shape}, scores={caption_scores.shape}")
    logger.info(f"Score range: [{caption_scores.min():.4f}, {caption_scores.max():.4f}]")
    logger.info(f"Mean score: {caption_scores.mean():.4f}")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info("=" * 60)
    
    # =========================================================
    # Example Output
    # =========================================================
    logger.info("\n--- Example: First 3 images and their neighbors ---")
    for img_idx in range(min(3, n_images)):
        image_id = image_ids_ordered[img_idx]
        cap_indices = image_to_indices[image_id]
        first_cap_idx = cap_indices[0]
        
        caption = dataset.samples[first_cap_idx]['caption']
        logger.info(f"\nImage [{img_idx}] (id={image_id}, {len(cap_indices)} captions)")
        logger.info(f"  Caption[{first_cap_idx}]: \"{caption[:60]}...\"")
        
        for k in range(min(3, top_k)):
            neighbor_cap_idx = caption_indices[first_cap_idx, k].item()
            neighbor_score = caption_scores[first_cap_idx, k].item()
            neighbor_caption = dataset.samples[neighbor_cap_idx]['caption']
            neighbor_image_id = dataset.samples[neighbor_cap_idx]['image_id']
            
            logger.info(f"  [{k+1}] Neighbor image_id={neighbor_image_id} (score={neighbor_score:.4f})")
            logger.info(f"       Caption[{neighbor_cap_idx}]: \"{neighbor_caption[:50]}...\"")
    
    # Verify mapping consistency
    logger.info("\n--- Verification: All captions of an image get same targets ---")
    test_image_id = image_ids_ordered[0]
    test_cap_indices = image_to_indices[test_image_id]
    
    first_targets = caption_indices[test_cap_indices[0]].tolist()
    all_same = all(
        caption_indices[cap_idx].tolist() == first_targets
        for cap_idx in test_cap_indices
    )
    logger.info(f"  Image {test_image_id} has {len(test_cap_indices)} captions")
    logger.info(f"  All captions share same targets: {all_same} ✓" if all_same else "  WARNING: Targets differ!")
    
    wandb.finish()
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
