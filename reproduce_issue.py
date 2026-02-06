
import sys
import os
sys.path.append(os.getcwd())

from src.data import create_image_text_dataloader
from unittest.mock import MagicMock

def test_truncation():
    # Mock components
    tokenizer = MagicMock()
    
    # Mock config with debug mode enabled and float debug_samples (to test the fix for floats too)
    config = {
        'data': {
            'images_path': '/tmp',
            'captions_path': '/tmp/captions.json',
            'batch_size': 2,
            'num_workers': 0
        },
        'training': {
            'batch_size': 2
        },
        'debug': {
            'debug_mode': True,
            'debug_samples': 5
        }
    }
    
    # Create a dummy captions file
    import json
    dummy_data = {
        'images': [
            {'split': 'train', 'cocoid': i, 'sentences': [{'raw': f'cap {i}'}], 'filename': f'img{i}.jpg'}
            for i in range(10)
        ]
    }
    with open('/tmp/captions.json', 'w') as f:
        json.dump(dummy_data, f)
        
    try:
        # This will trigger the truncation logic
        loader = create_image_text_dataloader(config, tokenizer, split='train')
        print(f"Loader created successfully. Dataset size: {len(loader.dataset)}")
        
        # Verify truncation happened
        if len(loader.dataset) == 5:
            print("Truncation works correctly.")
        else:
            print(f"Truncation FAILED. Size: {len(loader.dataset)}")
            
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_truncation()
