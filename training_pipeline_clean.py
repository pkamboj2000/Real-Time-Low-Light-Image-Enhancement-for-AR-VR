import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms
from compact_unet import CompactUNet
from enhancement_vit import EnhancementViT
import json
import time
from typing import Dict, Tuple

class LOLDataset(Dataset):
    """LOL Dataset for low-light image enhancement"""
    def __init__(self, data_dir, mode='train', img_size=256):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_size = img_size
        
        # Set up paths
        if mode == 'train':
            self.low_dir = self.data_dir / 'low'
            self.high_dir = self.data_dir / 'high'
        else:
            self.low_dir = self.data_dir / 'low'
            self.high_dir = self.data_dir / 'high'
        
        # Get image pairs
        self.image_pairs = []
        if self.low_dir.exists() and self.high_dir.exists():
            low_images = list(self.low_dir.glob('*.png'))
            for low_path in low_images:
                high_path = self.high_dir / low_path.name
                if high_path.exists():
                    self.image_pairs.append((low_path, high_path))
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.image_pairs[idx]
        
        # Load images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply transforms
        low_img = self.transform(low_img)
        high_img = self.transform(high_img)
        
        return low_img, high_img

class TrainingPipeline:
    """Complete training pipeline for enhancement models"""
    def __init__(self, model_type='unet', img_size=256, device=None):
        self.model_type = model_type
        self.img_size = img_size
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Create model
        if model_type == 'unet':
            self.model = CompactUNet(in_channels=3, out_channels=3, features=32)
        elif model_type == 'vit':
            self.model = EnhancementViT(img_size=img_size, patch_size=16, embed_dim=384, depth=6)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (low_imgs, high_imgs) in enumerate(dataloader):
            low_imgs = low_imgs.to(self.device)
            high_imgs = high_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(low_imgs)
            loss = self.criterion(outputs, high_imgs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for low_imgs, high_imgs in dataloader:
                low_imgs = low_imgs.to(self.device)
                high_imgs = high_imgs.to(self.device)
                
                outputs = self.model(low_imgs)
                loss = self.criterion(outputs, high_imgs)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def train(self, train_dataset, val_dataset=None, epochs=50, batch_size=8):
        """Complete training loop"""
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None
        
        print(f"Training {self.model_type} model on {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = 0
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f'best_{self.model_type}_model.pth')
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            if val_loader:
                print(f'  Val Loss: {val_loss:.6f}')
            print(f'  Time: {epoch_time:.2f}s')
            print('-' * 50)
        
        # Save final model
        self.save_model(f'final_{self.model_type}_model.pth')
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_type': self.model_type,
            'img_size': self.img_size,
            'device': str(self.device)
        }
        
        with open(f'{self.model_type}_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    def save_model(self, filename):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'img_size': self.img_size
        }, filename)
        print(f'Model saved: {filename}')

def create_synthetic_dataset(num_samples=1000, img_size=256):
    """Create synthetic low-light dataset for testing"""
    print("Creating synthetic dataset...")
    
    # Create directory
    data_dir = Path('synthetic_data')
    (data_dir / 'low').mkdir(parents=True, exist_ok=True)
    (data_dir / 'high').mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create high-quality image
        high_img = np.random.rand(img_size, img_size, 3) * 255
        high_img = high_img.astype(np.uint8)
        
        # Create low-light version
        brightness_factor = np.random.uniform(0.1, 0.4)
        low_img = (high_img.astype(np.float32) * brightness_factor).astype(np.uint8)
        
        # Add noise to low-light image
        noise = np.random.normal(0, 5, low_img.shape)
        low_img = np.clip(low_img + noise, 0, 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(str(data_dir / 'high' / f'{i:04d}.png'), high_img)
        cv2.imwrite(str(data_dir / 'low' / f'{i:04d}.png'), low_img)
    
    print(f"Created {num_samples} synthetic image pairs")
    return data_dir

def main():
    """Main training function"""
    print("Real-Time Low-Light Image Enhancement Training Pipeline")
    print("=" * 60)
    
    # Configuration
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 20
    
    # Create synthetic dataset
    data_dir = create_synthetic_dataset(num_samples=500, img_size=IMG_SIZE)
    
    # Create datasets
    train_dataset = LOLDataset(data_dir, mode='train', img_size=IMG_SIZE)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Train U-Net
    print("\nTraining U-Net Model...")
    unet_trainer = TrainingPipeline(model_type='unet', img_size=IMG_SIZE)
    unet_trainer.train(train_dataset, val_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Train ViT
    print("\nTraining ViT Model...")
    vit_trainer = TrainingPipeline(model_type='vit', img_size=IMG_SIZE)
    vit_trainer.train(train_dataset, val_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("\nTraining completed!")
    print("Models saved:")
    print("- best_unet_model.pth")
    print("- final_unet_model.pth")
    print("- best_vit_model.pth")
    print("- final_vit_model.pth")

if __name__ == '__main__':
    main()
