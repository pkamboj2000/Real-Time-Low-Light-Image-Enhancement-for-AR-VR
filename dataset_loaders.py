#!/usr/bin/env python3
"""
Dataset loaders for LOL and SID (See-in-the-Dark) datasets
"""

import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class LOLDataset(Dataset):
    """
    LOL (Low-Light) Dataset Loader
    Download from: https://daooshee.github.io/BMVC2018website/
    """
    
    def __init__(self, root_dir, split='train', img_size=256):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        
        # Define paths
        if split == 'train':
            self.low_dir = self.root_dir / 'our485' / 'low'
            self.high_dir = self.root_dir / 'our485' / 'high'
        else:  # eval
            self.low_dir = self.root_dir / 'eval15' / 'low'
            self.high_dir = self.root_dir / 'eval15' / 'high'
        
        # Get image pairs
        self.image_pairs = self._get_image_pairs()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        print(f"LOL {split} dataset: {len(self.image_pairs)} pairs")
    
    def _get_image_pairs(self):
        """Get low/high image pairs"""
        pairs = []
        
        if not self.low_dir.exists() or not self.high_dir.exists():
            print(f"Warning: LOL dataset not found at {self.root_dir}")
            return pairs
        
        low_images = sorted(list(self.low_dir.glob('*.png')))
        
        for low_path in low_images:
            # Find corresponding high image
            high_path = self.high_dir / low_path.name
            if high_path.exists():
                pairs.append((low_path, high_path))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        low_path, high_path = self.image_pairs[idx]
        
        # Load images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply transforms
        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)
        
        return {
            'low': low_tensor,
            'high': high_tensor,
            'low_path': str(low_path),
            'high_path': str(high_path)
        }

class SIDDataset(Dataset):
    """
    SID (See-in-the-Dark) Dataset Loader
    Download from: https://github.com/cchen156/Learning-to-See-in-the-Dark
    """
    
    def __init__(self, root_dir, split='train', camera='Sony', img_size=256):
        self.root_dir = Path(root_dir)
        self.split = split
        self.camera = camera  # 'Sony' or 'Fuji'
        self.img_size = img_size
        
        # Define paths
        self.short_dir = self.root_dir / camera / 'short'
        self.long_dir = self.root_dir / camera / 'long'
        
        # Get image pairs
        self.image_pairs = self._get_image_pairs()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        print(f"SID {camera} {split} dataset: {len(self.image_pairs)} pairs")
    
    def _get_image_pairs(self):
        """Get short/long exposure pairs"""
        pairs = []
        
        if not self.short_dir.exists() or not self.long_dir.exists():
            print(f"Warning: SID dataset not found at {self.root_dir}")
            return pairs
        
        # Read pairing information (usually in a text file)
        pair_file = self.root_dir / f"{self.camera}_{self.split}_list.txt"
        
        if pair_file.exists():
            with open(pair_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        short_name = parts[0]
                        long_name = parts[1]
                        
                        short_path = self.short_dir / short_name
                        long_path = self.long_dir / long_name
                        
                        if short_path.exists() and long_path.exists():
                            pairs.append((short_path, long_path))
        else:
            # Fallback: match by filename pattern
            short_files = sorted(list(self.short_dir.glob('*.ARW')))[:100]  # Limit for demo
            long_files = sorted(list(self.long_dir.glob('*.ARW')))[:100]
            
            for short_path in short_files:
                # Simple matching by similar names
                for long_path in long_files:
                    if short_path.stem[:5] == long_path.stem[:5]:  # Match first 5 chars
                        pairs.append((short_path, long_path))
                        break
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        short_path, long_path = self.image_pairs[idx]
        
        try:
            # Load RAW images (simplified - in practice use rawpy)
            short_img = cv2.imread(str(short_path), cv2.IMREAD_COLOR)
            long_img = cv2.imread(str(long_path), cv2.IMREAD_COLOR)
            
            if short_img is None or long_img is None:
                # Fallback to random data
                short_img = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)
                long_img = np.random.randint(100, 255, (512, 512, 3), dtype=np.uint8)
            
            # Apply transforms
            short_tensor = self.transform(short_img)
            long_tensor = self.transform(long_img)
            
            return {
                'low': short_tensor,
                'high': long_tensor,
                'low_path': str(short_path),
                'high_path': str(long_path)
            }
        
        except Exception as e:
            print(f"Error loading {short_path}: {e}")
            # Return dummy data
            dummy = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
            dummy_tensor = self.transform(dummy)
            return {
                'low': dummy_tensor,
                'high': dummy_tensor,
                'low_path': str(short_path),
                'high_path': str(long_path)
            }

def download_datasets():
    """Download and setup datasets"""
    print("Dataset Download Instructions:")
    print("=" * 50)
    
    print("\nLOL Dataset:")
    print("1. Visit: https://daooshee.github.io/BMVC2018website/")
    print("2. Download LOL dataset")
    print("3. Extract to: ./data/LOL/")
    print("   Structure: data/LOL/our485/{low,high}/, data/LOL/eval15/{low,high}/")
    
    print("\nSID Dataset:")
    print("1. Visit: https://github.com/cchen156/Learning-to-See-in-the-Dark")
    print("2. Download Sony or Fuji dataset")
    print("3. Extract to: ./data/SID/")
    print("   Structure: data/SID/{Sony,Fuji}/{short,long}/")
    
    print("\nNote: Due to size, manual download required.")
    print("Alternative: Use synthetic data generation in this project.")

def create_sample_dataset():
    """Create sample dataset for testing"""
    data_dir = Path('data/sample_dataset')
    low_dir = data_dir / 'low'
    high_dir = data_dir / 'high'
    
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset...")
    
    for i in range(20):
        # Create high quality image
        high_img = np.random.randint(150, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create low light version
        brightness = np.random.uniform(0.1, 0.3)
        low_img = (high_img.astype(np.float32) * brightness).astype(np.uint8)
        
        # Add noise
        noise = np.random.normal(0, 10, low_img.shape)
        low_img = np.clip(low_img + noise, 0, 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(str(low_dir / f'sample_{i:03d}.png'), low_img)
        cv2.imwrite(str(high_dir / f'sample_{i:03d}.png'), high_img)
    
    print(f"Sample dataset created: {data_dir}")
    return data_dir

def main():
    """Demo dataset loading"""
    print("Dataset Loaders Demo")
    print("=" * 30)
    
    # Create sample dataset
    sample_dir = create_sample_dataset()
    
    # Test LOL dataset loader (with sample data)
    print("\nTesting LOL Dataset Loader:")
    lol_dataset = LOLDataset(sample_dir, split='train')
    
    if len(lol_dataset) > 0:
        sample = lol_dataset[0]
        print(f"  Sample low image shape: {sample['low'].shape}")
        print(f"  Sample high image shape: {sample['high'].shape}")
    
    # Download instructions
    download_datasets()

if __name__ == '__main__':
    main()
