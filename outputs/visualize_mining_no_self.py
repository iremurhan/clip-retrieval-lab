#!/usr/bin/env python3
"""
visualize_mining_no_self.py
---------------------------
Visualizes mining results EXCLUDING the query image itself (Top-1).
This reveals the distribution of 'Hard Negatives' and 'Semantic Neighbors'.
"""

import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def load_scores_exclude_self(filepath):
    """
    Loads scores and excludes the first column (Top-1), 
    assuming it is the query image itself.
    Returns flattened array of Top-2 to Top-50 scores.
    """
    try:
        data = torch.load(filepath, map_location="cpu")
        scores = None

        # Case 1: Dictionary format
        if isinstance(data, dict) and 'scores' in data:
            scores = data['scores']
        
        # Case 2: Tuple/List format
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            scores = data[1]

        if scores is None:
            return None

        # Ensure it's a numpy array
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()
        else:
            scores = np.array(scores)

        # --- CRITICAL STEP: EXCLUDE SELF ---
        # scores shape is expected to be [N_images, TopK] (e.g., 113287, 50)
        # We slice [:, 1:] to remove the first column (index 0).
        if len(scores.shape) == 2 and scores.shape[1] > 1:
            scores_no_self = scores[:, 1:] 
            return scores_no_self.flatten()
        else:
            print(f"[WARNING] Shape {scores.shape} usually implies 1D. Cannot exclude self robustly.")
            return scores.flatten()

    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize mining results (Self-Excluded)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .pt files')
    parser.add_argument('--output', type=str, default=None, help='Output PDF path')
    
    args = parser.parse_args()
    input_dir = os.path.expanduser(args.input_dir)
    
    # Auto-detect dataset name
    dataset_name = 'Unknown'
    if 'flickr' in input_dir.lower(): dataset_name = 'Flickr30k'
    elif 'coco' in input_dir.lower(): dataset_name = 'COCO'

    if args.output:
        output_pdf = args.output
    else:
        output_pdf = os.path.join(input_dir, f"{dataset_name}_Mining_Report_NO_SELF.pdf")

    print(f"Dataset: {dataset_name} (Self-Excluded Analysis)")
    print(f"Reading from: {input_dir}")

    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    if not pt_files:
        print("No .pt files found.")
        return

    with PdfPages(output_pdf) as pdf:
        # Title Page
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Mining Analysis (Self-Excluded)\n\n{dataset_name}\n(Top-1 Removed)', 
                 ha='center', va='center', fontsize=20)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Histograms
        for fpath in sorted(pt_files):
            filename = os.path.basename(fpath)
            print(f"Processing: {filename}...")
            
            scores = load_scores_exclude_self(fpath)
            if scores is None: continue

            plt.figure(figsize=(10, 6))
            
            # Histogram
            # Using log scale often helps see the 'tail' of hard negatives
            plt.hist(scores, bins=100, color='salmon', edgecolor='black', alpha=0.7)
            
            # Stats
            stats = (f"Mean: {scores.mean():.4f}\n"
                     f"Std:  {scores.std():.4f}\n"
                     f"Max:  {scores.max():.4f} (Best Neighbor)\n"
                     f"Min:  {scores.min():.4f}\n"
                     f">0.8: {(scores>0.8).mean()*100:.2f}%\n"
                     f">0.7: {(scores>0.7).mean()*100:.2f}%")

            plt.title(f"{filename} (Neighbors Only)", fontsize=14)
            plt.xlabel("Similarity Score (Cosine)", fontsize=12)
            plt.ylabel("Frequency (Log Scale)", fontsize=12)
            plt.yscale('log')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, 
                     fontsize=10, va='top', ha='right', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            pdf.savefig()
            plt.close()

        # Comparative Plot
        plt.figure(figsize=(12, 7))
        for fpath in sorted(pt_files):
            scores = load_scores_exclude_self(fpath)
            if scores is None: continue
            
            label = os.path.basename(fpath).replace(".pt", "").replace("mining_", "")
            counts, bins = np.histogram(scores, bins=100, density=True)
            plt.plot((bins[:-1]+bins[1:])/2, counts, label=label, linewidth=2)

        plt.title("Neighbor Similarity Distribution (No Self)", fontsize=16)
        plt.xlabel("Similarity", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()

    print(f"\n[DONE] Report saved to: {output_pdf}")

if __name__ == "__main__":
    main()