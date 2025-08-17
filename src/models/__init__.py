"""
Model initialization and factory functions.
"""

import torch
from .unet import UNet, LightweightUNet, ResidualUNet, create_unet_model
from .vision_transformer import VisionTransformer, LightweightViT, create_vit_model
from ..utils.device import get_optimal_device, optimize_for_device

__all__ = [
    'UNet',
    'LightweightUNet', 
    'ResidualUNet',
    'VisionTransformer',
    'LightweightViT',
    'create_unet_model',
    'create_vit_model',
    'create_model'
]


def create_model(model_type='unet', variant='standard', device=None, **kwargs):
    """
    Factory function to create any supported model with automatic device optimization.
    
    Args:
        model_type (str): Type of model ('unet', 'vit')
        variant (str): Model variant ('standard', 'lightweight', 'residual')
        device (str or torch.device): Target device (auto-detected if None)
        **kwargs: Additional arguments for model initialization
    
    Returns:
        torch.nn.Module: Model instance optimized for target device
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_optimal_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Create the model
    if model_type == 'unet':
        model = create_unet_model(variant, **kwargs)
    elif model_type == 'vit':
        model = create_vit_model(variant, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Optimize for target device
    model = optimize_for_device(model, device)
    
    return model


def get_model_info(model):
    """
    Get information about a model.
    
    Args:
        model (torch.nn.Module): Model instance
    
    Returns:
        dict: Model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
