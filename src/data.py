"""
src/data.py
-----------
Data loading and preprocessing for Cross-Modal Retrieval.
Handles dataset loading for Flickr30k and COCO using standard CLIP transforms.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Configure logger
logger = logging.getLogger(__name__)

class CaptionImageDataset(Dataset):
    def __init__(
        self,
        images_root_path,
        captions_path,
        tokenizer,
        max_length=77,
        split='train',
        transform=None,
        mining_indices_path=None,
        mining_values_path=None,
        consensus_path=None,
        fne_threshold=0.90,
    ):
        """
        Args:
            images_root_path (str): Path to image folder.
            captions_path (str): Path to Karpathy JSON file.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Token sequence length (Default: 77).
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Image transforms.
            mining_indices_path (str, optional): Path to .pt file of neighbor indices [N, K].
            mining_values_path (str, optional): Path to .pt file of neighbor similarity scores [N, K].
            consensus_path (str, optional): Path to consensus .pt file (Image-Image).
            fne_threshold (float): Score above which a neighbor is treated as false negative (default 0.90).
        """
        self.images_root_path = images_root_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.fne_threshold = fne_threshold
        self.mining_indices = None
        self.mining_values = None
        
        # Load Text-Text Mining Data
        if mining_indices_path and mining_values_path and os.path.isfile(mining_indices_path) and os.path.isfile(mining_values_path):
            self.mining_indices = torch.load(mining_indices_path, map_location='cpu')
            self.mining_values = torch.load(mining_values_path, map_location='cpu')
            logger.info(f"Mining loaded: indices {mining_indices_path}, values {mining_values_path}, fne_threshold={fne_threshold}")

        # Load Consensus Mining Data (Image-Image)
        self.consensus_data = None
        self.img_id_to_consensus_idx = None
        if consensus_path and os.path.isfile(consensus_path):
            logger.info(f"Loading Consensus Data from {consensus_path}...")
            consensus_data = torch.load(consensus_path, map_location='cpu')
            # Expected format: {'mode': 'id_mapping', 'image_ids': tensor, 'indices': tensor, 'scores': tensor}
            self.consensus_image_ids = consensus_data['image_ids']
            self.consensus_indices = consensus_data['indices']
            self.consensus_scores = consensus_data['scores']
            
            # Map image_id -> row index in consensus tensors
            # Using a dict for O(1) lookup. 
            self.img_id_to_consensus_idx = {
                img_id.item(): idx for idx, img_id in enumerate(self.consensus_image_ids)
            }
            self.consensus_active = True
            logger.info(f"Consensus Data Loaded: {len(self.consensus_image_ids)} images.")
        else:
            self.consensus_active = False

        # 1. Define Transforms (Standard CLIP Preprocessing)
        if transform is None:
            # CLIP normalization values
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
            
            if split == 'train':
                # Standard training transform (RandomResizedCrop is standard for training stability)
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(336, scale=(0.9, 1.0)), # Mild augmentation
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
                # Separate augmentation transform for intra-modal consistency
                self.transform_aug = transforms.Compose([
                    transforms.RandomResizedCrop(336, scale=(0.9, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                # Validation/Test: Deterministic CenterCrop
                self.transform = transforms.Compose([
                    transforms.Resize(336, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(336),
                    transforms.ToTensor(),
                    normalize,
                ])
                self.transform_aug = self.transform  # Same for val/test
        else:
            self.transform = transform
            self.transform_aug = transform

        # 2. Load Captions (Karpathy JSON)
        logger.info(f"Loading captions from {captions_path} for split: {split}")
        with open(captions_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        
        # Karpathy JSON parsing logic
        for img in data['images']:
            current_split = img['split']
            if current_split == 'restval' and split == 'train':
                current_split = 'train'

            if current_split == split:
                # Handle ID variations (COCO: cocoid/id, Flickr30k: imgid)
                if 'cocoid' in img:
                    img_id = int(img['cocoid'])
                elif 'imgid' in img:
                    img_id = int(img['imgid'])
                elif 'id' in img:
                    img_id = int(img['id'])
                else:
                    raise ValueError(
                        f"Image entry missing 'cocoid', 'imgid', or 'id'. Keys: {list(img.keys())}"
                    )
                
                # Take exactly 5 captions
                sentences = img['sentences'][:5]
                for sent in sentences:
                    self.samples.append({
                        'image_id': img_id,
                        'caption': sent['raw'],
                        'filepath': img.get('filepath', ''),
                        'filename': img.get('filename', '')
                    })
                    
        logger.info(f"Found {len(self.samples)} samples for split '{split}'.")
        # Map sample index -> image_id for FNE (exclude same-image negatives)
        self.caption_to_img_id = [s['image_id'] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']
        
        # 1. Load Image
        filepath = sample.get('filepath', '').strip()
        filename = sample['filename']
        
        if filepath:
            image_path = os.path.join(self.images_root_path, filepath, filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(self.images_root_path, filename)
        else:
            image_path = os.path.join(self.images_root_path, filename)
        
        # Safety check for missing images
        if not os.path.exists(image_path):
            # Fallback strategy? Or raise error? For now, let's error out to notice data issues.
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        # 2. Transform Image (original and augmented for intra-modal loss)
        img_tensor = self.transform(image)
        img_aug_tensor = self.transform_aug(image)
        
        # 3. Tokenize Text
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        out = {
            'image': img_tensor,
            'image_aug': img_aug_tensor,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'image_id': image_id
        }

        # 4. False Negative Elimination (FNE): optional hard negative from mining
        if self.mining_indices is not None and self.mining_values is not None:
            # Neighbors for this index: shape [K] or [1]
            indices = self.mining_indices[idx]
            values = self.mining_values[idx]
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
                values = values.unsqueeze(0)
            
            # Determine valid negatives based on FNE strategy
            valid_mask = torch.ones_like(indices, dtype=torch.bool)
            
            # 1. Base check: Must differ by image ID
            candidate_img_ids = [self.caption_to_img_id[i] for i in indices.tolist()]
            # This check is always required
            valid_mask &= (torch.tensor(candidate_img_ids, dtype=torch.long) != image_id)

            if self.consensus_active:
                # --- CONSENSUS-BASED FNE ---
                # Check if candidate negative is a "consensus neighbor" of the anchor image
                
                # Check anchor presence
                if image_id not in self.img_id_to_consensus_idx:
                    raise KeyError(f"Anchor image_id {image_id} not found in Consensus Data!")
                    
                anchor_consensus_idx = self.img_id_to_consensus_idx[image_id]
                
                # Get anchor's consensus neighbors (indices into consensus_image_ids)
                # shape: [K_consensus]
                anchor_neighbor_indices = self.consensus_indices[anchor_consensus_idx] 
                anchor_neighbor_scores = self.consensus_scores[anchor_consensus_idx]
                
                # We need to check if each candidate negative image corresponds to a high-score neighbor
                for i, cand_img_id in enumerate(candidate_img_ids):
                    if not valid_mask[i]:
                        continue # Already invalid
                        
                    if cand_img_id not in self.img_id_to_consensus_idx:
                         raise KeyError(f"Candidate image_id {cand_img_id} not found in Consensus Data!")
                    
                    cand_consensus_idx = self.img_id_to_consensus_idx[cand_img_id]
                    
                    # Check if cand_consensus_idx is in anchor_neighbor_indices
                    # Note: We assume consensus_indices[anchor_idx] contains top-k most similar images.
                    # Find if cand_consensus_idx exists in the neighbor list using torch.where or similar
                    match = (anchor_neighbor_indices == cand_consensus_idx).nonzero(as_tuple=True)[0]
                    
                    if len(match) > 0:
                        # It is a neighbor. Check score.
                        score = anchor_neighbor_scores[match[0]]
                        if score > self.fne_threshold:
                            # It's a False Negative (too similar) -> Mask it out
                            valid_mask[i] = False
                    # If not in top-k neighbors, we verify it's safe (implicitly score < low_threshold of top-k)
                    
            else:
                # --- FACTBACK: TEXT-TEXT FNE ---
                # Valid negatives: score <= fne_threshold
                valid_mask &= (values <= self.fne_threshold)

            valid_indices = indices[valid_mask].tolist()

            if valid_indices:
                # Hard negative from mined neighbors (different image, below FNE threshold)
                neg_idx = random.choice(valid_indices)
                neg_caption = self.samples[neg_idx]['caption']
            else:
                # No valid hard negative from mining → sample an easy negative:
                # randomly choose another caption from a DIFFERENT image.
                max_tries = 100
                neg_idx = None
                for _ in range(max_tries):
                    candidate_idx = random.randrange(len(self.samples))
                    if self.caption_to_img_id[candidate_idx] != image_id:
                        neg_idx = candidate_idx
                        break

                if neg_idx is None:
                    # Dataset is likely degenerate (single image); surface the issue explicitly.
                    raise RuntimeError(
                        f"Failed to sample an easy negative for index {idx}: "
                        "could not find a caption from a different image_id."
                    )

                neg_caption = self.samples[neg_idx]['caption']
            neg_tokenized = self.tokenizer(
                neg_caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            out['negative_input_ids'] = neg_tokenized['input_ids'].squeeze(0)
            out['negative_attention_mask'] = neg_tokenized['attention_mask'].squeeze(0)

        return out


def create_image_text_dataloader(config, tokenizer, split='train'):
    """
    Factory function to create DataLoader for image-text retrieval datasets.
    
    Creates a DataLoader for Flickr30k or COCO datasets with appropriate
    transforms, shuffling, and batch size based on the split.
    When split is 'train' and config['mining'] exists, passes mining paths and fne_threshold for FNE.
    """
    shuffle = (split == 'train')
    images_root = config['data']['images_path']
    captions_path = config['data']['captions_path']

    mining_kwargs = {}
    if split == 'train' and config.get('mining') is not None and config['mining'].get('enabled', False):
        mining_cfg = config['mining']
        mining_kwargs = {
            'mining_indices_path': mining_cfg.get('indices_path'),
            'mining_values_path': mining_cfg.get('values_path'),
            'fne_threshold': mining_cfg.get('fne_threshold', 0.90),
        }

    dataset = CaptionImageDataset(
        images_root_path=images_root,
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_length=config['data'].get('max_length', 77),
        split=split,
        **mining_kwargs
    )

    # Debug Truncation
    if config.get('debug', {}).get('debug_mode', False):
        debug_limit = config['debug'].get('debug_samples', 100)
        if len(dataset.samples) > debug_limit:
            logger.warning(f"DEBUG MODE: Truncating dataset to {debug_limit} samples.")
            dataset.samples = dataset.samples[:debug_limit]

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'], # Base Config yapısına uygun path
        shuffle=shuffle,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return loader