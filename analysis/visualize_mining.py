#!/usr/bin/env python3
"""
visualize_mining.py
-------------------
Visualize mining results from pairwise similarity mining.
Supports both COCO and Flickr30k datasets.
"""

import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def load_pt_file(filepath):
    """
    Loads the .pt file and extracts the score tensor.
    Handles different saving formats:
    - New format: {'mode': 'id_mapping'/'index_mapping', 'scores': ..., ...}
    - Legacy format: {'scores': ..., ...} or tuple/list
    """
    try:
        data = torch.load(filepath, map_location="cpu")
        
        # Case 1: New format with 'mode' key
        if isinstance(data, dict):
            if 'scores' in data:
                scores = data['scores']
                if isinstance(scores, torch.Tensor):
                    return scores.flatten().numpy()
                else:
                    return np.array(scores).flatten()
        
        # Case 2: Legacy tuple/list format
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            scores = data[1]
            if isinstance(scores, torch.Tensor):
                return scores.flatten().numpy()
            else:
                return np.array(scores).flatten()
        
        print(f"[WARNING] Could not parse format for: {filepath}")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize mining results from .pt files')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .pt mining files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name for title (optional)')
    
    args = parser.parse_args()
    
    input_dir = os.path.expanduser(args.input_dir)
    
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return
    
    # Determine dataset name automatically if not provided
    dataset_name = args.dataset
    if not dataset_name:
        if 'flickr' in input_dir.lower():
            dataset_name = 'Flickr30k'
        elif 'coco' in input_dir.lower():
            dataset_name = 'COCO'
        else:
            dataset_name = 'Unknown Dataset'
    
    # Determine output path
    if args.output:
        output_pdf = args.output
    else:
        output_pdf = os.path.join(input_dir, f"{dataset_name}_mining_analysis_report.pdf")
    
    print(f"Starting analysis in directory: {input_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Output PDF: {output_pdf}")
    
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    if not pt_files:
        print(f"[ERROR] No .pt files found in: {input_dir}")
        return
    
    print(f"Found {len(pt_files)} mining files.")
    
    with PdfPages(output_pdf) as pdf:
        
        # Page 1: Title Page
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Mining Results Analysis Report\n\n{dataset_name} Dataset', 
                 horizontalalignment='center', verticalalignment='center', fontsize=20)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Page 2+: Histograms
        for fpath in sorted(pt_files):
            filename = os.path.basename(fpath)
            print(f"Processing: {filename}...")
            
            scores = load_pt_file(fpath)
            if scores is None or len(scores) == 0:
                print(f"  [SKIP] Could not load scores from {filename}")
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Histogram
            plt.hist(scores, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            
            # --- DETAILED STATISTICS ---
            stats_lines = [
                f"Min: {scores.min():.4f}",
                f"Max: {scores.max():.4f}",
                f"Mean: {scores.mean():.4f}",
                f"Std: {scores.std():.4f}",
                f"Median: {np.median(scores):.4f}",
                "-" * 20,
                f"5th %: {np.percentile(scores, 5):.4f}",
                f"95th %: {np.percentile(scores, 95):.4f}"
            ]
            
            # Threshold Ratios (0.7, 0.8, 0.9)
            stats_lines.append("-" * 20)
            for thresh in [0.7, 0.8, 0.9]:
                ratio = (scores > thresh).mean() * 100
                stats_lines.append(f"> {thresh:.1f}: {ratio:.2f}%")
            
            stats_text = "\n".join(stats_lines)
            
            # Plot Settings
            plt.title(f"Score Distribution: {filename}", fontsize=14)
            plt.xlabel("Similarity Score (Cosine Sim)", fontsize=12)
            plt.ylabel("Frequency (Log Scale)", fontsize=12)
            plt.yscale('log')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Add stats box (Adjusted fontsize to fit more info)
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                     fontsize=8, verticalalignment='top', horizontalalignment='right',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            pdf.savefig()
            plt.close()

        # Last Page: Comparative Plot
        print("Generating comparative plot...")
        plt.figure(figsize=(12, 7))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        valid_files = []
        for fpath in sorted(pt_files):
            scores = load_pt_file(fpath)
            if scores is not None and len(scores) > 0:
                valid_files.append((fpath, scores))
        
        if valid_files:
            for i, (fpath, scores) in enumerate(valid_files):
                label = os.path.basename(fpath).replace(".pt", "").replace("mining_", "").replace("image_", "")
                counts, bins = np.histogram(scores, bins=100, density=True)
                centers = (bins[:-1] + bins[1:]) / 2
                plt.plot(centers, counts, label=label, linewidth=2, color=colors[i % len(colors)])

            plt.title("Comparison of Similarity Distributions", fontsize=16)
            plt.xlabel("Similarity Score", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            pdf.savefig()
            plt.close()

    print(f"\n[SUCCESS] Report generated: {output_pdf}")

if __name__ == "__main__":
    main()