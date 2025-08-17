"""
Device detection and management utilities.

Provides automatic GPU/MPS detection and device optimization for training and inference.

Author: Pranjal Kamboj
Created: August 2025
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_optimal_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the optimal computation device based on availability.
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU over CPU when available
        
    Returns:
        torch.device: Optimal device for computation
    """
    if prefer_gpu:
        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("Using Apple Silicon MPS backend")
            return torch.device('mps')
        
        # Check for CUDA
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA backend - {torch.cuda.get_device_name()}")
            return device
    
    # Fallback to CPU
    logger.info("Using CPU backend")
    return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.
    
    Returns:
        dict: Device information including availability and specs
    """
    info = {
        'cpu': {
            'available': True,
            'cores': torch.get_num_threads()
        },
        'cuda': {
            'available': torch.cuda.is_available(),
            'devices': []
        },
        'mps': {
            'available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'built': torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
        }
    }
    
    # CUDA device info
    if info['cuda']['available']:
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info['cuda']['devices'].append({
                'id': i,
                'name': device_props.name,
                'memory_gb': device_props.total_memory / (1024**3),
                'compute_capability': f"{device_props.major}.{device_props.minor}"
            })
    
    return info


def optimize_for_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Optimize model for specific device.
    
    Args:
        model (torch.nn.Module): Model to optimize
        device (torch.device): Target device
        
    Returns:
        torch.nn.Module: Optimized model
    """
    model = model.to(device)
    
    # Device-specific optimizations
    if device.type == 'cuda':
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("Enabled cuDNN optimizations for CUDA")
        
    elif device.type == 'mps':
        # MPS-specific optimizations
        logger.info("Applied MPS optimizations")
        
    return model


def synchronize_device(device: torch.device):
    """
    Synchronize computation on specified device.
    
    Args:
        device (torch.device): Device to synchronize
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def get_memory_usage(device: torch.device) -> dict:
    """
    Get memory usage statistics for device.
    
    Args:
        device (torch.device): Device to check
        
    Returns:
        dict: Memory usage information
    """
    if device.type == 'cuda':
        return {
            'allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(device) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024**3)
        }
    else:
        return {'message': f'Memory stats not available for {device.type}'}


def clear_cache(device: torch.device):
    """
    Clear device cache to free up memory.
    
    Args:
        device (torch.device): Device to clear cache for
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    elif device.type == 'mps':
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache")


def print_device_summary():
    """Print a summary of available devices and recommendations."""
    info = get_device_info()
    optimal_device = get_optimal_device()
    
    print("\n🔧 Device Summary")
    print("=" * 50)
    
    # CPU info
    print(f"CPU: ✅ Available ({info['cpu']['cores']} threads)")
    
    # CUDA info
    if info['cuda']['available']:
        print(f"CUDA: ✅ Available")
        for device_info in info['cuda']['devices']:
            print(f"  - {device_info['name']} ({device_info['memory_gb']:.1f}GB)")
    else:
        print("CUDA: ❌ Not available")
    
    # MPS info
    if info['mps']['available'] and info['mps']['built']:
        print("MPS: ✅ Available (Apple Silicon)")
    else:
        print("MPS: ❌ Not available")
    
    print(f"\n🎯 Recommended device: {optimal_device}")
    print("=" * 50)


if __name__ == "__main__":
    print_device_summary()
