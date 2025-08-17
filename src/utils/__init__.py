"""
Utility functions for the project.
"""

import os
import json
import yaml
import numpy as np
import torch
from typing import Dict, Any, Union
import logging
from pathlib import Path


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        
    Returns:
        logging.Logger: Configured logger
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_results(results: Dict[str, Any], results_path: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        results_path: Path to save results
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def get_device(device: str = 'auto') -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: PyTorch device
    """
    if device == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 * 1024)  # Assuming float32


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        self.elapsed = 0
    
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if self.start is not None:
            self.start.record()
        else:
            import time
            self.start_time = time.perf_counter()
        
        return self
    
    def __exit__(self, *args):
        if self.end is not None:
            self.end.record()
            torch.cuda.synchronize()
            self.elapsed = self.start.elapsed_time(self.end) / 1000  # Convert to seconds
        else:
            import time
            self.elapsed = time.perf_counter() - self.start_time


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test formatting
    print(f"Time formatting: {format_time(3661.5)}")
    print(f"Size formatting: {format_size(1536000000)}")
    
    print("Utilities working correctly!")
