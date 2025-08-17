"""
Dataset handling and preprocessing utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LowLightDataset(Dataset):
    """
    Dataset class for low-light image enhancement.
    Supports paired and unpaired training data.
    """
    
    def __init__(
        self,
        low_light_dir: str,
        normal_light_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        mode: str = 'train',
        augmentation: bool = True,
        paired: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            low_light_dir (str): Directory containing low-light images
            normal_light_dir (str, optional): Directory containing normal-light images
            image_size (tuple): Target image size (height, width)
            mode (str): Dataset mode ('train', 'val', 'test')
            augmentation (bool): Whether to apply data augmentation
            paired (bool): Whether the dataset has paired images
        """
        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.image_size = image_size
        self.mode = mode
        self.paired = paired
        
        # Get image paths
        self.low_light_paths = self._get_image_paths(low_light_dir)
        
        if paired and normal_light_dir:
            self.normal_light_paths = self._get_image_paths(normal_light_dir)
            # Ensure paired images have same count
            assert len(self.low_light_paths) == len(self.normal_light_paths), \
                "Number of low-light and normal-light images must match for paired training"
        else:
            self.normal_light_paths = None
        
        # Setup transformations
        self.transform = self._get_transforms(augmentation)
        
    def _get_image_paths(self, directory: str) -> List[str]:
        """Get all image paths from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        paths = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    paths.append(os.path.join(root, file))
        
        return sorted(paths)
    
    def _get_transforms(self, augmentation: bool) -> A.Compose:
        """Setup image transformations."""
        transforms_list = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ]
        
        if augmentation and self.mode == 'train':
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
            ])
        
        transforms_list.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        return A.Compose(transforms_list)
    
    def __len__(self) -> int:
        return len(self.low_light_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            dict: Dictionary containing 'low_light' and optionally 'normal_light' tensors
        """
        # Load low-light image
        low_light_path = self.low_light_paths[idx]
        low_light_image = cv2.imread(low_light_path)
        low_light_image = cv2.cvtColor(low_light_image, cv2.COLOR_BGR2RGB)
        
        sample = {'low_light': low_light_image}
        
        # Load normal-light image if available
        if self.paired and self.normal_light_paths:
            normal_light_path = self.normal_light_paths[idx]
            normal_light_image = cv2.imread(normal_light_path)
            normal_light_image = cv2.cvtColor(normal_light_image, cv2.COLOR_BGR2RGB)
            sample['normal_light'] = normal_light_image
        
        # Apply transformations
        if self.paired and 'normal_light' in sample:
            # Apply same transformations to both images
            transformed = self.transform(
                image=sample['low_light'],
                mask=sample['normal_light']
            )
            return {
                'low_light': transformed['image'],
                'normal_light': transformed['mask'],
                'path': low_light_path
            }
        else:
            transformed = self.transform(image=sample['low_light'])
            return {
                'low_light': transformed['image'],
                'path': low_light_path
            }


class Noise2NoiseDataset(Dataset):
    """
    Dataset for Noise2Noise self-supervised training.
    Creates two noisy versions of each clean image.
    """
    
    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        noise_type: str = 'gaussian',
        noise_level: float = 0.1
    ):
        """
        Initialize Noise2Noise dataset.
        
        Args:
            image_dir (str): Directory containing clean images
            image_size (tuple): Target image size
            noise_type (str): Type of noise ('gaussian', 'poisson')
            noise_level (float): Noise intensity
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        self.image_paths = self._get_image_paths(image_dir)
        
        # Basic transforms
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def _get_image_paths(self, directory: str) -> List[str]:
        """Get all image paths from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        paths = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    paths.append(os.path.join(root, file))
        
        return sorted(paths)
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise to image."""
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_level, image.shape)
            noisy = image + noise
        elif self.noise_type == 'poisson':
            # Poisson noise
            noisy = np.random.poisson(image / self.noise_level) * self.noise_level
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return np.clip(noisy, 0, 1)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get noisy image pair for Noise2Noise training."""
        image_path = self.image_paths[idx]
        
        # Load and normalize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Create two noisy versions
        noisy1 = self._add_noise(image)
        noisy2 = self._add_noise(image)
        
        # Convert back to uint8 for transforms
        noisy1 = (noisy1 * 255).astype(np.uint8)
        noisy2 = (noisy2 * 255).astype(np.uint8)
        
        # Apply transforms
        transformed1 = self.transform(image=noisy1)
        transformed2 = self.transform(image=noisy2)
        
        return {
            'input': transformed1['image'],
            'target': transformed2['image'],
            'path': image_path
        }


def create_data_loaders(
    train_low_dir: str,
    val_low_dir: str,
    train_normal_dir: Optional[str] = None,
    val_normal_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    paired: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_low_dir (str): Training low-light images directory
        val_low_dir (str): Validation low-light images directory
        train_normal_dir (str, optional): Training normal-light images directory
        val_normal_dir (str, optional): Validation normal-light images directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        paired (bool): Whether to use paired training
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = LowLightDataset(
        low_light_dir=train_low_dir,
        normal_light_dir=train_normal_dir,
        image_size=image_size,
        mode='train',
        augmentation=True,
        paired=paired
    )
    
    val_dataset = LowLightDataset(
        low_light_dir=val_low_dir,
        normal_light_dir=val_normal_dir,
        image_size=image_size,
        mode='val',
        augmentation=False,
        paired=paired
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_noise2noise_loader(
    image_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    noise_type: str = 'gaussian',
    noise_level: float = 0.1
) -> DataLoader:
    """
    Create data loader for Noise2Noise training.
    
    Args:
        image_dir (str): Directory containing clean images
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        noise_type (str): Type of noise to add
        noise_level (float): Noise intensity
    
    Returns:
        DataLoader: Data loader for Noise2Noise training
    """
    dataset = Noise2NoiseDataset(
        image_dir=image_dir,
        image_size=image_size,
        noise_type=noise_type,
        noise_level=noise_level
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset creation...")
    
    # This would require actual data directories
    # train_loader, val_loader = create_data_loaders(
    #     train_low_dir="data/processed/LOL/train/low",
    #     val_low_dir="data/processed/LOL/test/low",
    #     train_normal_dir="data/processed/LOL/train/high",
    #     val_normal_dir="data/processed/LOL/test/high",
    #     batch_size=4
    # )
    
    print("Dataset classes created successfully!")
