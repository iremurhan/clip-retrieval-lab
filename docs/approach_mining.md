# Semantic Hard Negative Mining with False Negative Elimination

## Problem
In standard triplet loss, "hard negatives" are crucial for learning. However, in large-scale datasets (like Flickr30k/COCO), many distinct images have semantically identical captions (e.g., "A dog running on grass"). 

If we blindly select the closest retrieval neighbors as negatives, we introduce **False Negatives** (penalizing the model for retrieving a semantically correct image that just happens to have a different ID).

## Approach

We utilize an offline mining strategy with a safety filter:

1.  **Mining:** We pre-compute image-text or text-text similarity using a frozen CLIP model to find the top-k nearest neighbors for every anchor.
2.  **Filtering (False Negative Elimination):** During triplet generation, we filter candidates based on two criteria:
    * **ID Check:** The negative must not share the same Image ID as the anchor.
    * **Semantic Threshold:** If the similarity score between the anchor and the candidate is above a hyperparameter `mining_threshold` (e.g., 0.90), we discard it.
    
**Hypothesis:** Candidates with similarity `> threshold` are likely synonyms (False Negatives). Candidates with similarity `< threshold` but still in the top-k are valid Hard Negatives.

## Artifacts
* **Indices:** `mining_text_top1000_indices.pt`
* **Values:** `mining_text_top1000_values.pt`