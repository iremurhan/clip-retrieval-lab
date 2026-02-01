import os
import json
import argparse
import torch
from tqdm import tqdm

def load_dataset_json(json_path, split='train'):
    """Dataset JSON dosyasını yükler ve caption-image eşleşmelerini çıkarır."""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Karpathy split formatına göre filtrele
    if split == 'train':
        images = [img for img in data['images'] if img['split'] == 'train' or img.get('restval', False)]
    else:
        images = [img for img in data['images'] if img['split'] == split]
    
    caption_list = []      # Tüm caption stringleri
    caption_to_imgid = []  # caption_index -> imgid
    imgid_to_filename = {} # imgid -> filename
    
    print(f"Processing {len(images)} images for split '{split}'...")
    
    for img in images:
        img_id = img.get('imgid', img.get('id'))
        filename = img['filename']
        imgid_to_filename[img_id] = filename
        
        for sent in img['sentences']:
            caption_list.append(sent['raw'])
            caption_to_imgid.append(img_id)
            
    return caption_list, caption_to_imgid, imgid_to_filename

def main():
    parser = argparse.ArgumentParser(description="Create triplets with False Negative Elimination")
    parser.add_argument('--dataset_json', type=str, required=True)
    parser.add_argument('--mining_indices', type=str, required=True, help='Path to .pt indices file')
    parser.add_argument('--mining_values', type=str, required=True, help='Path to .pt values (scores) file')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_negatives', type=int, default=1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--threshold', type=float, default=0.90, help='Similarity threshold. Scores above this are ignored (False Negatives).')
    
    args = parser.parse_args()

    # 1. Dataset Yükle
    captions, caption_to_imgid, imgid_to_filename = load_dataset_json(args.dataset_json, args.split)
    
    # 2. Mining Dosyalarını Yükle (Indices ve Values)
    print(f"Loading indices: {args.mining_indices}")
    print(f"Loading values:  {args.mining_values}")
    
    try:
        mining_indices = torch.load(args.mining_indices, map_location='cpu')
        mining_values = torch.load(args.mining_values, map_location='cpu')
    except Exception as e:
        print(f"Error loading .pt files: {e}")
        return

    if len(captions) != mining_indices.shape[0]:
        print(f"WARNING: Size mismatch! Dataset: {len(captions)}, Mining: {mining_indices.shape[0]}")

    triplets = []
    skipped_false_negatives = 0 # İstatistik için sayaç
    
    print(f"Generating triplets (Negatives: {args.num_negatives}, Max Threshold: {args.threshold})...")
    
    # 3. Triplet Döngüsü
    for idx, (caption_text, pos_img_id) in enumerate(tqdm(zip(captions, caption_to_imgid), total=len(captions))):
        
        neighbors = mining_indices[idx]
        scores = mining_values[idx]
        
        hard_negatives = []
        
        for i, neg_idx in enumerate(neighbors):
            neg_idx = neg_idx.item()
            score = scores[i].item()
            
            # 1. Kendisi mi?
            if neg_idx == idx:
                continue
            
            # 2. Aynı resme mi ait? (ID Kontrolü)
            neg_img_id = caption_to_imgid[neg_idx]
            if neg_img_id == pos_img_id:
                continue
            
            # 3. Skoru çok mu yüksek? (Threshold Kontrolü - YENİ KISIM)
            # Eğer skor > threshold ise, bu cümle muhtemelen 'True Positive' gibidir.
            # Negatif olarak kullanırsak modele zarar veririz. Atlıyoruz.
            if score > args.threshold:
                skipped_false_negatives += 1
                continue
            
            # Buraya geldiyse: Farklı resim AND skor threshold'un altında (Valid Hard Negative)
            hard_negatives.append(captions[neg_idx])
            
            if len(hard_negatives) >= args.num_negatives:
                break
        
        if len(hard_negatives) > 0:
            triplet = {
                "image_filename": imgid_to_filename[pos_img_id],
                "caption": caption_text,
                "negatives": hard_negatives
            }
            triplets.append(triplet)

    print(f"Skipped {skipped_false_negatives} candidates because similarity > {args.threshold}")
    print(f"Saving {len(triplets)} triplets to {args.output_path}...")
    
    with open(args.output_path, 'w') as f:
        json.dump(triplets, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    main()