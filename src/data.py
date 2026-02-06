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
    ):
        """
        Args:
            images_root_path (str): Path to image folder.
            captions_path (str): Path to Karpathy JSON file.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Token sequence length (Default: 77).
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Image transforms.
        """
        self.images_root_path = images_root_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Define Transforms (Standard CLIP Preprocessing)
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

        # Load Captions (Karpathy JSON)
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
        logger.info(f"Found {len(self.samples)} samples for split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']
        
        # Load Image
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

        # Transform Image (original and augmented for intra-modal loss)
        img_tensor = self.transform(image)
        img_aug_tensor = self.transform_aug(image)
        
        # Tokenize Text
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

    images_root = config['data']['images_path']
    captions_path = config['data']['captions_path']

    dataset = CaptionImageDataset(
        images_root_path=images_root,
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_length=config['data'].get('max_length', 77),
        split=split,
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