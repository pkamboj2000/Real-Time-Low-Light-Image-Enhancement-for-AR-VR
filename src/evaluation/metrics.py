"""
Image quality evaluation metrics for enhancement assessment.

Comprehensive metrics suite including PSNR, SSIM, LPIPS and performance
benchmarking for real-time applications.

Author: Pranjal Kamboj
Created: August 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional, Union
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available - install with: pip install lpips")

try:
    from pytorch_msssim import ssim, ms_ssim
    PYTORCH_SSIM_AVAILABLE = True
except ImportError:
    PYTORCH_SSIM_AVAILABLE = False
    print("pytorch-msssim not available - using skimage SSIM implementation")


class ImageQualityMetrics:
    """
    Comprehensive image quality assessment for enhancement evaluation.
    
    Provides multiple metrics to evaluate both technical quality (PSNR, SSIM)
    and perceptual quality (LPIPS) along with performance benchmarking.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize metrics computation engine.
        
        Args:
            device (str): Computation device ('cpu', 'cuda', 'mps')
        """
        self.device = device
        
        # Initialize LPIPS model for perceptual evaluation
        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
                print(f"LPIPS model loaded on {device}")
            except Exception as e:
                self.lpips_model = None
                print(f"Failed to load LPIPS model: {e}")
        else:
            self.lpips_model = None
    
    def psnr(self, img1: Union[np.ndarray, torch.Tensor], 
             img2: Union[np.ndarray, torch.Tensor], 
             data_range: float = 255.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between two images.
        
        Classic metric for image quality assessment. Higher values indicate
        better quality and less noise.
        
        Args:
            img1: Reference image
            img2: Enhanced/test image  
            data_range: Maximum possible pixel value (255 for 8-bit images)
            
        Returns:
            float: PSNR value in dB (higher is better)
        """
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        # Ensure same shape
        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
        
        # Calculate PSNR
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
        return float(psnr_value)
    
    def ssim(self, img1: Union[np.ndarray, torch.Tensor], 
             img2: Union[np.ndarray, torch.Tensor],
             data_range: float = 255.0,
             multichannel: bool = True) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            img1: First image
            img2: Second image
            data_range: Maximum possible pixel value
            multichannel: Whether to treat as multichannel image
            
        Returns:
            float: SSIM value
        """
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        # Handle different input formats
        if img1.ndim == 4:  # Batch dimension
            img1 = img1[0]
            img2 = img2[0]
        
        if img1.ndim == 3 and img1.shape[0] == 3:  # CHW format
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        ssim_value = structural_similarity(
            img1, img2, 
            data_range=data_range,
            multichannel=multichannel,
            channel_axis=-1 if multichannel else None
        )
        
        return float(ssim_value)
    
    def lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS).
        
        Args:
            img1: First image tensor (B, C, H, W) in range [0, 1]
            img2: Second image tensor (B, C, H, W) in range [0, 1]
            
        Returns:
            float: LPIPS distance
        """
        if self.lpips_model is None:
            raise RuntimeError("LPIPS model not available")
        
        # Ensure tensors are on correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Normalize to [-1, 1] range expected by LPIPS
        img1_norm = 2 * img1 - 1
        img2_norm = 2 * img2 - 1
        
        with torch.no_grad():
            lpips_value = self.lpips_model(img1_norm, img2_norm).mean()
        
        return float(lpips_value.cpu())
    
    def pytorch_ssim(self, img1: torch.Tensor, img2: torch.Tensor, 
                     data_range: float = 1.0) -> float:
        """
        Calculate SSIM using PyTorch implementation.
        
        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
            data_range: Maximum possible pixel value
            
        Returns:
            float: SSIM value
        """
        ssim_value = ssim(img1, img2, data_range=data_range)
        return float(ssim_value)
    
    def ms_ssim(self, img1: torch.Tensor, img2: torch.Tensor, 
                data_range: float = 1.0) -> float:
        """
        Calculate Multi-Scale SSIM.
        
        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
            data_range: Maximum possible pixel value
            
        Returns:
            float: MS-SSIM value
        """
        ms_ssim_value = ms_ssim(img1, img2, data_range=data_range)
        return float(ms_ssim_value)


class PerformanceMetrics:
    """Performance and efficiency metrics."""
    
    def __init__(self):
        self.timings = []
        self.memory_usage = []
    
    def measure_inference_time(self, model: torch.nn.Module, 
                             input_tensor: torch.Tensor, 
                             num_runs: int = 100,
                             warmup_runs: int = 10) -> Dict[str, float]:
        """
        Measure model inference time.
        
        Args:
            model: Model to evaluate
            input_tensor: Input tensor
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            dict: Timing statistics
        """
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize before timing (important for GPU)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                # Synchronize after inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times)),
            'ms_per_frame': float(np.mean(times) * 1000)
        }
    
    def measure_memory_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Measure model memory usage.
        
        Args:
            model: Model to evaluate
            
        Returns:
            dict: Memory usage statistics
        """
        device = next(model.parameters()).device
        
        # Calculate model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = param_size + buffer_size
        
        result = {
            'model_size_mb': model_size / (1024 * 1024),
            'param_size_mb': param_size / (1024 * 1024),
            'buffer_size_mb': buffer_size / (1024 * 1024)
        }
        
        # GPU memory usage (if available)
        if device.type == 'cuda':
            result['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            result['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        elif device.type == 'mps':
            result['mps_memory_allocated_mb'] = torch.mps.current_allocated_memory() / (1024 * 1024)
        
        return result


class EvaluationSuite:
    """Complete evaluation suite for image enhancement models."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize evaluation suite.
        
        Args:
            device: Device to run evaluations on
        """
        self.device = device
        self.quality_metrics = ImageQualityMetrics(device)
        self.performance_metrics = PerformanceMetrics()
    
    def evaluate_model(self, model: torch.nn.Module, 
                      test_loader: torch.utils.data.DataLoader,
                      save_examples: bool = True,
                      num_examples: int = 5) -> Dict[str, any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            save_examples: Whether to save example outputs
            num_examples: Number of examples to save
            
        Returns:
            dict: Evaluation results
        """
        model.eval()
        model = model.to(self.device)
        
        results = {
            'quality_metrics': [],
            'performance_metrics': {},
            'examples': []
        }
        
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # Move to device
                low_light = batch['low_light'].to(self.device)
                
                # Check if we have ground truth
                has_ground_truth = 'normal_light' in batch
                
                if has_ground_truth:
                    ground_truth = batch['normal_light'].to(self.device)
                
                # Measure inference time for first batch
                if i == 0:
                    timing_results = self.performance_metrics.measure_inference_time(
                        model, low_light[:1], num_runs=50
                    )
                    results['performance_metrics']['timing'] = timing_results
                
                # Generate enhanced image
                enhanced = model(low_light)
                
                # Calculate metrics if ground truth available
                if has_ground_truth:
                    batch_size = low_light.shape[0]
                    
                    for j in range(batch_size):
                        # Convert to numpy for some metrics
                        enhanced_np = enhanced[j].cpu().numpy()
                        gt_np = ground_truth[j].cpu().numpy()
                        
                        # Convert from CHW to HWC and scale to [0, 255]
                        if enhanced_np.shape[0] == 3:
                            enhanced_np = np.transpose(enhanced_np, (1, 2, 0))
                            gt_np = np.transpose(gt_np, (1, 2, 0))
                        
                        enhanced_np = (enhanced_np * 255).astype(np.uint8)
                        gt_np = (gt_np * 255).astype(np.uint8)
                        
                        # Calculate metrics
                        psnr_val = self.quality_metrics.psnr(enhanced_np, gt_np)
                        ssim_val = self.quality_metrics.ssim(enhanced_np, gt_np)
                        
                        total_psnr += psnr_val
                        total_ssim += ssim_val
                        total_samples += 1
                        
                        # LPIPS (on tensors)
                        if self.quality_metrics.lpips_model is not None:
                            lpips_val = self.quality_metrics.lpips(
                                enhanced[j:j+1], ground_truth[j:j+1]
                            )
                            total_lpips += lpips_val
                
                # Save examples
                if save_examples and len(results['examples']) < num_examples:
                    example = {
                        'low_light': low_light[0].cpu(),
                        'enhanced': enhanced[0].cpu(),
                        'path': batch['path'][0] if 'path' in batch else f'example_{i}'
                    }
                    if has_ground_truth:
                        example['ground_truth'] = ground_truth[0].cpu()
                    
                    results['examples'].append(example)
        
        # Calculate average metrics
        if total_samples > 0:
            results['quality_metrics'] = {
                'psnr': total_psnr / total_samples,
                'ssim': total_ssim / total_samples,
                'lpips': total_lpips / total_samples if self.quality_metrics.lpips_model else None,
                'num_samples': total_samples
            }
        
        # Memory usage
        results['performance_metrics']['memory'] = self.performance_metrics.measure_memory_usage(model)
        
        return results
    
    def print_results(self, results: Dict[str, any]):
        """Print evaluation results in a formatted way."""
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Quality metrics
        if results['quality_metrics']:
            quality = results['quality_metrics']
            print("\n📊 QUALITY METRICS:")
            print(f"  PSNR:  {quality['psnr']:.2f} dB")
            print(f"  SSIM:  {quality['ssim']:.4f}")
            if quality['lpips'] is not None:
                print(f"  LPIPS: {quality['lpips']:.4f}")
            print(f"  Samples: {quality['num_samples']}")
        
        # Performance metrics
        if 'timing' in results['performance_metrics']:
            timing = results['performance_metrics']['timing']
            print("\n⚡ PERFORMANCE METRICS:")
            print(f"  FPS: {timing['fps']:.1f}")
            print(f"  Latency: {timing['ms_per_frame']:.1f} ms/frame")
            print(f"  Mean time: {timing['mean_time']*1000:.1f} ± {timing['std_time']*1000:.1f} ms")
        
        if 'memory' in results['performance_metrics']:
            memory = results['performance_metrics']['memory']
            print("\n💾 MEMORY USAGE:")
            print(f"  Model size: {memory['model_size_mb']:.1f} MB")
            if 'gpu_memory_allocated_mb' in memory:
                print(f"  GPU memory: {memory['gpu_memory_allocated_mb']:.1f} MB")
            elif 'mps_memory_allocated_mb' in memory:
                print(f"  MPS memory: {memory['mps_memory_allocated_mb']:.1f} MB")
        
        print("=" * 60)


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create dummy images
    img1 = np.random.rand(256, 256, 3) * 255
    img2 = img1 + np.random.rand(256, 256, 3) * 10  # Add some noise
    
    metrics = ImageQualityMetrics()
    
    psnr_val = metrics.psnr(img1, img2)
    ssim_val = metrics.ssim(img1, img2)
    
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    
    print("Evaluation metrics ready!")
