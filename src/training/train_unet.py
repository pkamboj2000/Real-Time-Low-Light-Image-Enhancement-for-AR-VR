"""
U-Net training script for low-light image enhancement.

Supports both supervised learning with paired data and self-supervised 
Noise2Noise training for unpaired datasets.

Author: Pranjal Kamboj
Created: August 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import sys
from tqdm import tqdm
import numpy as np
import wandb
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import create_model
from data.dataset import create_data_loaders, create_noise2noise_loader
from evaluation.metrics import ImageQualityMetrics
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Combined loss function for image enhancement training.
    
    Combines L1 loss for pixel accuracy with perceptual and structural losses
    for better visual quality. Loss weights were tuned empirically.
    """
    
    def __init__(self, l1_weight=0.8, perceptual_weight=0.2, ssim_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        # Initialize LPIPS model for perceptual loss
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex')
            print("LPIPS perceptual loss initialized successfully")
        except ImportError:
            self.lpips_model = None
            print("Warning: LPIPS not available. Install with: pip install lpips")
    
    def forward(self, pred, target):
        # L1 loss for pixel-wise accuracy
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM loss for structural similarity
        ssim_loss = 1 - self.ssim(pred, target)
        
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        # Perceptual loss (if available)
        if self.lpips_model is not None and self.perceptual_weight > 0:
            # Normalize to [-1, 1] for LPIPS
            pred_norm = 2 * pred - 1
            target_norm = 2 * target - 1
            perceptual_loss = self.lpips_model(pred_norm, target_norm).mean()
            total_loss += self.perceptual_weight * perceptual_loss
        
        return total_loss
    
    def ssim(self, img1, img2):
        """Simple SSIM implementation."""
        # Convert to grayscale if needed
        if img1.shape[1] == 3:
            img1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            img2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
            img1 = img1.unsqueeze(1)
            img2 = img2.unsqueeze(1)
        
        # SSIM parameters
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class Trainer:
    """Training class for image enhancement models."""
    
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        
        # Create model
        self.model = create_model(**config['model']).to(self.device)
        
        # Create loss function
        self.criterion = CombinedLoss(**config['loss'])
        if self.criterion.lpips_model is not None:
            self.criterion.lpips_model = self.criterion.lpips_model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics = ImageQualityMetrics(self.device)
        
        # Initialize logging
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                config=config
            )
            wandb.watch(self.model)
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_metric = -float('inf') if config['checkpoint']['mode'] == 'max' else float('inf')
        self.patience_counter = 0
    
    def _get_device(self):
        """Get training device."""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_config)
    
    def _create_optimizer(self):
        """Create optimizer."""
        opt_config = self.config['training']
        
        if opt_config['optimizer'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config['training']['scheduler']
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            low_light = batch['low_light'].to(self.device)
            
            if 'normal_light' in batch:
                # Supervised training
                target = batch['normal_light'].to(self.device)
            else:
                # Self-supervised (Noise2Noise)
                target = batch['target'].to(self.device)
                low_light = batch['input'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config['hardware']['mixed_precision']:
                with torch.autocast(device_type=self.device.type):
                    enhanced = self.model(low_light)
                    loss = self.criterion(enhanced, target)
                
                # Backward pass with gradient scaling
                scaler = torch.cuda.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                enhanced = self.model(low_light)
                loss = self.criterion(enhanced, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to wandb
            if (self.config['logging']['use_wandb'] and 
                batch_idx % self.config['logging']['log_every'] == 0):
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                low_light = batch['low_light'].to(self.device)
                
                if 'normal_light' in batch:
                    target = batch['normal_light'].to(self.device)
                else:
                    target = batch['target'].to(self.device)
                    low_light = batch['input'].to(self.device)
                
                # Forward pass
                enhanced = self.model(low_light)
                loss = self.criterion(enhanced, target)
                total_loss += loss.item()
                
                # Calculate metrics
                for i in range(enhanced.shape[0]):
                    enhanced_np = enhanced[i].cpu().numpy().transpose(1, 2, 0)
                    target_np = target[i].cpu().numpy().transpose(1, 2, 0)
                    
                    enhanced_np = (enhanced_np * 255).astype(np.uint8)
                    target_np = (target_np * 255).astype(np.uint8)
                    
                    psnr = self.metrics.psnr(enhanced_np, target_np)
                    ssim = self.metrics.ssim(enhanced_np, target_np)
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    num_samples += 1
        
        avg_loss = total_loss / len(val_loader)
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        return {
            'val_loss': avg_loss,
            'val_psnr': avg_psnr,
            'val_ssim': avg_ssim
        }
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint']['save_dir'],
            'latest_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint']['save_dir'],
                'best_checkpoint.pth'
            )
            torch.save(checkpoint, best_path)
        
        # Save epoch checkpoint
        if (self.epoch + 1) % self.config['checkpoint']['save_every'] == 0:
            epoch_path = os.path.join(
                self.config['checkpoint']['save_dir'],
                f'checkpoint_epoch_{self.epoch+1}.pth'
            )
            torch.save(checkpoint, epoch_path)
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Print metrics
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val PSNR: {val_metrics['val_psnr']:.2f}")
            print(f"Val SSIM: {val_metrics['val_ssim']:.4f}")
            print("-" * 50)
            
            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **val_metrics
                })
            
            # Check for best model
            monitor_metric = self.config['checkpoint']['monitor_metric']
            current_metric = val_metrics[monitor_metric]
            
            is_best = False
            if self.config['checkpoint']['mode'] == 'max':
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping']['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for low-light enhancement')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    if config['data'].get('noise2noise', False):
        train_loader = create_noise2noise_loader(
            config['data']['train_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size']
        )
        val_loader = create_noise2noise_loader(
            config['data']['val_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size']
        )
    else:
        train_loader, val_loader = create_data_loaders(
            train_low_dir=config['data']['train_low_dir'],
            val_low_dir=config['data']['val_low_dir'],
            train_normal_dir=config['data'].get('train_normal_dir'),
            val_normal_dir=config['data'].get('val_normal_dir'),
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size'],
            paired=config['data']['paired']
        )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {trainer.epoch}")
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
