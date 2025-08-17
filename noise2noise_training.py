import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms
from compact_unet import CompactUNet
import time
import json

class Noise2NoiseDataset(Dataset):
    """Noise2Noise dataset for self-supervised learning"""
    
    def __init__(self, data_dir, img_size=256, noise_types=['gaussian', 'poisson']):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.noise_types = noise_types
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_dir.glob(f'**/{ext}')))
        
        print(f"Found {len(self.image_paths)} images for Noise2Noise training")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def add_gaussian_noise(self, image, noise_level=0.1):
        """Add Gaussian noise"""
        noise = torch.randn_like(image) * noise_level
        return torch.clamp(image + noise, 0, 1)
    
    def add_poisson_noise(self, image, lam=30.0):
        """Add Poisson noise (photon noise)"""
        # Scale image to have higher values for Poisson
        scaled = image * lam
        noisy = torch.poisson(scaled) / lam
        return torch.clamp(noisy, 0, 1)
    
    def add_low_light_noise(self, image, brightness_factor=0.3, noise_level=0.05):
        """Add realistic low-light noise"""
        # Reduce brightness
        dark = image * brightness_factor
        
        # Add noise that's more prominent in dark regions
        noise_strength = noise_level * (1 - dark)  # More noise in darker areas
        noise = torch.randn_like(image) * noise_strength
        
        return torch.clamp(dark + noise, 0, 1)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Load clean image
            image = Image.open(img_path).convert('RGB')
            clean = self.transform(image)
            
            # Generate two different noisy versions
            noise_type = random.choice(self.noise_types)
            
            if noise_type == 'gaussian':
                noisy1 = self.add_gaussian_noise(clean, random.uniform(0.05, 0.15))
                noisy2 = self.add_gaussian_noise(clean, random.uniform(0.05, 0.15))
            elif noise_type == 'poisson':
                noisy1 = self.add_poisson_noise(clean, random.uniform(20, 50))
                noisy2 = self.add_poisson_noise(clean, random.uniform(20, 50))
            else:  # low_light
                noisy1 = self.add_low_light_noise(clean, random.uniform(0.2, 0.4), random.uniform(0.03, 0.08))
                noisy2 = self.add_low_light_noise(clean, random.uniform(0.2, 0.4), random.uniform(0.03, 0.08))
            
            return noisy1, noisy2, clean
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return random tensors as fallback
            return torch.rand(3, self.img_size, self.img_size), \
                   torch.rand(3, self.img_size, self.img_size), \
                   torch.rand(3, self.img_size, self.img_size)

class Noise2NoiseTrainer:
    """Noise2Noise training implementation"""
    
    def __init__(self, model_type='unet', img_size=256, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.img_size = img_size
        
        # Create model
        if model_type == 'unet':
            self.model = CompactUNet(in_channels=3, out_channels=3, features=32)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.clean_losses = []  # Loss against clean images
        
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def train_epoch(self, dataloader):
        """Train for one epoch using Noise2Noise approach"""
        self.model.train()
        total_loss = 0
        total_clean_loss = 0
        total_psnr = 0
        num_batches = 0
        
        for batch_idx, (noisy1, noisy2, clean) in enumerate(dataloader):
            noisy1 = noisy1.to(self.device)
            noisy2 = noisy2.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Noise2Noise: predict noisy2 from noisy1
            pred = self.model(noisy1)
            loss = self.criterion(pred, noisy2)
            
            # Also calculate loss against clean for monitoring
            clean_loss = self.criterion(pred, clean)
            psnr = self.calculate_psnr(pred, clean)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_clean_loss += clean_loss.item()
            total_psnr += psnr.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, "
                      f"N2N Loss: {loss.item():.4f}, "
                      f"Clean Loss: {clean_loss.item():.4f}, "
                      f"PSNR: {psnr.item():.2f}")
        
        return {
            'n2n_loss': total_loss / num_batches,
            'clean_loss': total_clean_loss / num_batches,
            'psnr': total_psnr / num_batches
        }
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_clean_loss = 0
        total_psnr = 0
        num_batches = 0
        
        with torch.no_grad():
            for noisy1, noisy2, clean in dataloader:
                noisy1 = noisy1.to(self.device)
                noisy2 = noisy2.to(self.device)
                clean = clean.to(self.device)
                
                pred = self.model(noisy1)
                loss = self.criterion(pred, noisy2)
                clean_loss = self.criterion(pred, clean)
                psnr = self.calculate_psnr(pred, clean)
                
                total_loss += loss.item()
                total_clean_loss += clean_loss.item()
                total_psnr += psnr.item()
                num_batches += 1
        
        return {
            'n2n_loss': total_loss / num_batches,
            'clean_loss': total_clean_loss / num_batches,
            'psnr': total_psnr / num_batches
        }
    
    def train(self, data_dir, epochs=30, batch_size=8, val_split=0.2):
        """Complete Noise2Noise training"""
        print(f"Starting Noise2Noise training on {self.device}")
        
        # Create dataset
        full_dataset = Noise2NoiseDataset(data_dir, self.img_size, 
                                         noise_types=['gaussian', 'poisson', 'low_light'])
        
        # Split dataset
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        best_clean_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['n2n_loss'])
            self.clean_losses.append(train_metrics['clean_loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['n2n_loss'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['n2n_loss'])
            
            # Save best model based on clean loss (actual denoising performance)
            if val_metrics['clean_loss'] < best_clean_loss:
                best_clean_loss = val_metrics['clean_loss']
                self.save_model('best_noise2noise_model.pth')
                print("💾 New best model saved!")
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
            print(f"  Train - N2N: {train_metrics['n2n_loss']:.4f}, "
                  f"Clean: {train_metrics['clean_loss']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}")
            print(f"  Val   - N2N: {val_metrics['n2n_loss']:.4f}, "
                  f"Clean: {val_metrics['clean_loss']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f}")
            print("-" * 60)
        
        # Save final model
        self.save_model('final_noise2noise_model.pth')
        
        # Save training history
        history = {
            'train_n2n_losses': self.train_losses,
            'val_n2n_losses': self.val_losses,
            'train_clean_losses': self.clean_losses,
            'epochs': epochs,
            'best_clean_loss': best_clean_loss,
            'device': str(self.device)
        }
        
        with open('noise2noise_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Noise2Noise training completed!")
        print(f"📊 Best clean loss: {best_clean_loss:.4f}")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'clean_losses': self.clean_losses
        }, filename)

def create_sample_data(num_images=200, img_size=256):
    """Create sample clean images for Noise2Noise training"""
    data_dir = Path('noise2noise_data')
    data_dir.mkdir(exist_ok=True)
    
    print(f"Creating {num_images} sample images...")
    
    for i in range(num_images):
        # Create diverse synthetic images
        if i % 4 == 0:
            # Smooth gradients
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            for c in range(3):
                x, y = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
                img[:, :, c] = (128 + 127 * np.sin(2 * np.pi * (x + y + c * 0.33))).astype(np.uint8)
        elif i % 4 == 1:
            # Random shapes
            img = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
            cv2.circle(img, (img_size//2, img_size//2), img_size//4, (255, 255, 255), -1)
        elif i % 4 == 2:
            # Texture patterns
            img = np.random.randint(100, 255, (img_size, img_size, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (15, 15), 5)
        else:
            # High frequency patterns
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        cv2.imwrite(str(data_dir / f'sample_{i:04d}.png'), img)
    
    print(f"Sample data created in {data_dir}")
    return data_dir

def main():
    """Demo Noise2Noise training"""
    print("🧠 Noise2Noise Self-Supervised Training")
    print("=" * 50)
    
    # Create sample data
    data_dir = create_sample_data(num_images=100, img_size=256)
    
    # Initialize trainer
    trainer = Noise2NoiseTrainer(model_type='unet', img_size=256)
    
    # Start training
    trainer.train(
        data_dir=data_dir,
        epochs=10,  # Quick demo
        batch_size=4,
        val_split=0.2
    )
    
    print("\n🎉 Noise2Noise training demo completed!")
    print("📁 Models saved:")
    print("  - best_noise2noise_model.pth")
    print("  - final_noise2noise_model.pth")
    print("  - noise2noise_training_history.json")

if __name__ == '__main__':
    main()
