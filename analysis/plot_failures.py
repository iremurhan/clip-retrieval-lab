import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def plot_failures(json_path, output_pdf):
    # JSON dosyasını yükle
    with open(json_path) as f:
        data = json.load(f)
    
    # Text-to-Image Hataları (En kritik olanlar)
    failures = data.get('t2i_failures', [])
    
    if not failures:
        print("Hata bulunamadı veya JSON formatı farklı.")
        return

    print(f"Raporlanacak hata sayısı: {len(failures)}")
    
    with PdfPages(output_pdf) as pdf:
        # Kapak Sayfası
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Failure Analysis Report\n(Hard Negatives)', 
                 ha='center', va='center', fontsize=24)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        for i, case in enumerate(failures[:50]): # İlk 50 hatayı göster
            query_text = case.get('query_caption', 'No Caption')
            gt_img_path = case.get('gt_image_path')
            pred_img_path = case.get('retrieved_image_path')
            rank = case.get('rank', -1)
            
            # Eğer pathler yoksa atla
            if not gt_img_path or not pred_img_path:
                continue
                
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Sol: Modelin Yanlış Tahmini
            try:
                ax[0].imshow(Image.open(pred_img_path))
                ax[0].set_title(f"MODEL PREDICTION (Wrong)\nRank: {rank}", color='red', fontsize=10)
            except:
                ax[0].text(0.5, 0.5, "Image Not Found", ha='center')
            ax[0].axis('off')
            
            # Sağ: Olması Gereken
            try:
                ax[1].imshow(Image.open(gt_img_path))
                ax[1].set_title(f"GROUND TRUTH\n(Correct Image)", color='green', fontsize=10)
            except:
                ax[1].text(0.5, 0.5, "Image Not Found", ha='center')
            ax[1].axis('off')
            
            plt.suptitle(f"Query: \"{query_text[:80]}...\"", fontsize=12, y=0.95)
            
            pdf.savefig()
            plt.close()
            print(f"Processed case {i+1}")

    print(f"PDF Raporu hazır: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to hard_negatives.json")
    parser.add_argument('--output', type=str, default="failure_report.pdf", help="Output PDF path")
    args = parser.parse_args()
    
    plot_failures(args.input, args.output)