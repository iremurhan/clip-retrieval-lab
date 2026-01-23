#!/usr/bin/env python3
"""
analyze_distributions.py
------------------------
Analyzes the Text-Text and Image-Image similarity distributions
of a trained model or raw CLIP model.

Supports both COCO and Flickr30k datasets.

Outputs:
1. Similarity Histograms (PNG)
2. Statistical Summary (JSON)
3. Raw Scores (JSON) - for further analysis

Usage:
    python tools/analyze_distributions.py --dataset_name coco --split test
    python tools/analyze_distributions.py --dataset_name flickr30k --split test
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CocoImageDataset


def setup_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_dataset_paths(config, dataset_name):
    """
    Get dataset paths based on dataset name.
    
    Args:
        config: Configuration dictionary
        dataset_name: 'coco' or 'flickr30k'
    
    Returns:
        images_root: Path to images directory
        captions_path: Path to captions JSON file
    """
    if dataset_name == 'coco':
        images_root = config['data']['images_path']
        captions_path = config['data']['captions_path']
    elif dataset_name == 'flickr30k':
        # Flickr30k paths - try config first, then use convention-based paths
        flickr_config = config.get('data_flickr30k', {})
        
        if flickr_config:
            images_root = flickr_config.get('images_path', 'datasets/flickr30k')
            captions_path = flickr_config.get('captions_path', 'datasets/flickr30k/dataset_flickr30k.json')
        else:
            # Convention: Replace 'coco' with 'flickr30k' in default paths
            base_images = config['data']['images_path']
            base_captions = config['data']['captions_path']
            
            images_root = base_images.replace('coco', 'flickr30k')
            captions_path = base_captions.replace('dataset_coco.json', 'dataset_flickr30k.json')
            captions_path = captions_path.replace('coco', 'flickr30k')
        
        print(f"[Flickr30k] Images: {images_root}")
        print(f"[Flickr30k] Captions: {captions_path}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return images_root, captions_path


def compute_all_embeddings(model, loader, device):
    """
    Extract embeddings for all images and texts in the dataset.
    
    Args:
        model: CLIP model or DualEncoder
        loader: DataLoader
        device: torch device
    
    Returns:
        img_embeds: Tensor of shape [N, D]
        txt_embeds: Tensor of shape [N, D]
    """
    model.eval()
    img_embeds = []
    txt_embeds = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # CLIP Forward (Vision & Text Encoder)
            # Supports both raw CLIP and fine-tuned DualEncoder
            if hasattr(model, 'get_image_features'):
                img_feat = model.get_image_features(pixel_values=images)
                txt_feat = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # DualEncoder structure
                img_feat, txt_feat = model(images, input_ids, attention_mask)
            
            # L2 Normalization (required for cosine similarity)
            img_feat = F.normalize(img_feat, p=2, dim=1)
            txt_feat = F.normalize(txt_feat, p=2, dim=1)
            
            img_embeds.append(img_feat.cpu())
            txt_embeds.append(txt_feat.cpu())
            
    return torch.cat(img_embeds), torch.cat(txt_embeds)


def compute_max_similarities(embeds, chunk_size=5000):
    """
    Compute max similarity for each sample (excluding self).
    Uses chunked computation to avoid OOM on large datasets.
    
    Args:
        embeds: Tensor of shape [N, D]
        chunk_size: Number of queries to process at once
    
    Returns:
        max_sims: Tensor of shape [N] containing max similarity for each sample
    """
    n = embeds.shape[0]
    max_sims = torch.zeros(n)
    
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = embeds[i:end]
        
        # Compute similarities for this chunk
        sims = torch.matmul(chunk, embeds.t())  # [chunk_size, N]
        
        # Mask self-similarity
        for j in range(end - i):
            sims[j, i + j] = -1.0
        
        # Get max for each query in chunk
        max_sims[i:end], _ = sims.max(dim=1)
    
    return max_sims


def main():
    parser = argparse.ArgumentParser(description='Analyze similarity distributions for COCO/Flickr30k')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset_name', type=str, default='coco', 
                        choices=['coco', 'flickr30k'], help='Dataset to analyze')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'], help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='analysis_results', 
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for feature extraction')
    parser.add_argument('--save_raw_scores', action='store_true', default=True,
                        help='Save raw similarity scores to JSON')
    args = parser.parse_args()

    setup_seed()

    # Load Config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[OK] Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[OK] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[WARN] Using CPU (no GPU acceleration)")

    # Load Model and Tokenizer
    model_name = config['model']['image_model_name']
    print(f"Loading model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Get dataset paths
    images_root, captions_path = get_dataset_paths(config, args.dataset_name)
    
    # Verify paths exist
    if not os.path.exists(images_root):
        print(f"[ERROR] Images path not found: {images_root}")
        print("Please ensure the dataset is downloaded and paths are correct.")
        sys.exit(1)
    
    if not os.path.exists(captions_path):
        print(f"[ERROR] Captions path not found: {captions_path}")
        print("Please ensure the dataset JSON file exists.")
        sys.exit(1)

    print(f"\nDataset: {args.dataset_name.upper()}")
    print(f"Split: {args.split}")
    print(f"Images: {images_root}")
    print(f"Captions: {captions_path}")

    # Load dataset
    dataset = CocoImageDataset(
        images_root_path=images_root,
        captions_path=captions_path,
        tokenizer=tokenizer,
        split=args.split
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=4,  # Reduced for stability
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 1. Compute Embeddings
    img_emb, txt_emb = compute_all_embeddings(model, loader, device)
    print(f"Embeddings shape: Images {img_emb.shape}, Texts {txt_emb.shape}")
    
    # 2. Compute Max Similarities (memory-efficient chunked computation)
    print("\nComputing Text-Text similarities...")
    max_sim_txt = compute_max_similarities(txt_emb)
    
    print("Computing Image-Image similarities...")
    max_sim_img = compute_max_similarities(img_emb)
    
    # 3. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. Visualization
    print("\nGenerating plots...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(max_sim_txt.numpy(), bins=100, color='blue', kde=True, alpha=0.7)
    plt.axvline(max_sim_txt.mean().item(), color='darkblue', linestyle='--', 
                label=f'Mean: {max_sim_txt.mean().item():.4f}')
    plt.title(f"{args.dataset_name.upper()} ({args.split}) - Text-Text Max Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.histplot(max_sim_img.numpy(), bins=100, color='red', kde=True, alpha=0.7)
    plt.axvline(max_sim_img.mean().item(), color='darkred', linestyle='--',
                label=f'Mean: {max_sim_img.mean().item():.4f}')
    plt.title(f"{args.dataset_name.upper()} ({args.split}) - Image-Image Max Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_dist.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")
    
    # 5. Compute and save statistics
    stats = {
        "dataset": args.dataset_name,
        "split": args.split,
        "num_images": int(img_emb.shape[0]),
        "num_captions": int(txt_emb.shape[0]),
        "embedding_dim": int(img_emb.shape[1]),
        "text_text_max_sim": {
            "mean": float(max_sim_txt.mean()),
            "std": float(max_sim_txt.std()),
            "min": float(max_sim_txt.min()),
            "max": float(max_sim_txt.max()),
            "median": float(max_sim_txt.median()),
            "p5": float(torch.quantile(max_sim_txt, 0.05)),
            "p95": float(torch.quantile(max_sim_txt, 0.95))
        },
        "image_image_max_sim": {
            "mean": float(max_sim_img.mean()),
            "std": float(max_sim_img.std()),
            "min": float(max_sim_img.min()),
            "max": float(max_sim_img.max()),
            "median": float(max_sim_img.median()),
            "p5": float(torch.quantile(max_sim_img, 0.05)),
            "p95": float(torch.quantile(max_sim_img, 0.95))
        }
    }
    
    stats_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved: {stats_path}")
    
    # 6. Save raw scores (for further analysis)
    if args.save_raw_scores:
        raw_scores = {
            "dataset": args.dataset_name,
            "split": args.split,
            "text_text_max_scores": max_sim_txt.tolist(),
            "image_image_max_scores": max_sim_img.tolist()
        }
        
        raw_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_raw_scores.json")
        with open(raw_path, 'w') as f:
            json.dump(raw_scores, f)
        print(f"Raw scores saved: {raw_path}")
    
    # 7. Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name.upper()} ({args.split})")
    print(f"Samples: {stats['num_images']} images, {stats['num_captions']} captions")
    print(f"\nText-Text Max Similarity:")
    print(f"  Mean: {stats['text_text_max_sim']['mean']:.4f} +/- {stats['text_text_max_sim']['std']:.4f}")
    print(f"  Range: [{stats['text_text_max_sim']['min']:.4f}, {stats['text_text_max_sim']['max']:.4f}]")
    print(f"  Percentiles (5th/95th): [{stats['text_text_max_sim']['p5']:.4f}, {stats['text_text_max_sim']['p95']:.4f}]")
    print(f"\nImage-Image Max Similarity:")
    print(f"  Mean: {stats['image_image_max_sim']['mean']:.4f} +/- {stats['image_image_max_sim']['std']:.4f}")
    print(f"  Range: [{stats['image_image_max_sim']['min']:.4f}, {stats['image_image_max_sim']['max']:.4f}]")
    print(f"  Percentiles (5th/95th): [{stats['image_image_max_sim']['p5']:.4f}, {stats['image_image_max_sim']['p95']:.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
