#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework
Implements PSNR, SSIM, LPIPS and performance benchmarking
"""

import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path
import json
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    psnr: float
    ssim: float
    processing_time: float
    memory_usage: float
    method_name: str

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = []
    
    def benchmark_method(self, method_func, input_image: np.ndarray, 
                        method_name: str, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark a single enhancement method"""
        times = []
        
        # Warm-up run
        _ = method_func(input_image.copy())
        
        # Benchmark runs
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = method_func(input_image.copy())
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }
    
    def calculate_image_metrics(self, enhanced: np.ndarray, 
                               reference: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics"""
        # Ensure images are in the same format
        if enhanced.dtype != np.float64:
            enhanced = enhanced.astype(np.float64) / 255.0
        if reference.dtype != np.float64:
            reference = reference.astype(np.float64) / 255.0
        
        # PSNR
        psnr_value = psnr(reference, enhanced, data_range=1.0)
        
        # SSIM
        if len(enhanced.shape) == 3:
            ssim_value = ssim(reference, enhanced, data_range=1.0, 
                            channel_axis=2, multichannel=True)
        else:
            ssim_value = ssim(reference, enhanced, data_range=1.0)
        
        # Simple perceptual metric (simplified LPIPS alternative)
        # Using gradient-based perceptual similarity
        def calculate_gradients(img):
            if len(img.shape) == 3:
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (img * 255).astype(np.uint8)
            
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(grad_x**2 + grad_y**2)
        
        grad_enhanced = calculate_gradients(enhanced)
        grad_reference = calculate_gradients(reference)
        
        # Normalized gradient difference as perceptual metric
        grad_diff = np.mean(np.abs(grad_enhanced - grad_reference))
        perceptual_similarity = 1.0 / (1.0 + grad_diff)
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'perceptual_similarity': perceptual_similarity
        }

class EvaluationFramework:
    """Comprehensive evaluation framework"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark = PerformanceBenchmark()
        self.results = []
    
    def load_test_images(self, test_dir: str) -> List[Tuple[np.ndarray, str]]:
        """Load test images from directory"""
        test_path = Path(test_dir)
        if not test_path.exists():
            print(f"Test directory {test_dir} not found. Creating sample images...")
            self.create_sample_test_images(test_dir)
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in test_path.glob(ext):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append((img, img_path.name))
        
        return images
    
    def create_sample_test_images(self, test_dir: str, num_images: int = 10):
        """Create sample test images"""
        test_path = Path(test_dir)
        test_path.mkdir(exist_ok=True)
        
        print(f"Creating {num_images} sample test images...")
        
        for i in range(num_images):
            # Create synthetic low-light image
            img = np.random.rand(256, 256, 3)
            
            # Add some structure
            x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
            pattern = np.sin(x * 10) * np.cos(y * 10)
            
            for c in range(3):
                img[:, :, c] = img[:, :, c] * 0.3 + pattern * 0.2 + 0.1
            
            # Make it darker (low-light simulation)
            img = np.clip(img * 0.4, 0, 1)
            
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.imwrite(str(test_path / f"test_image_{i:03d}.jpg"),
                       cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    def evaluate_method(self, method_func, method_name: str, 
                       test_images: List[Tuple[np.ndarray, str]], 
                       reference_images: List[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate a single enhancement method"""
        print(f"\nEvaluating {method_name}...")
        
        all_metrics = []
        performance_metrics = []
        
        for i, (test_img, img_name) in enumerate(test_images):
            print(f"Processing {img_name}...")
            
            # Performance benchmark
            perf_metrics = self.benchmark.benchmark_method(
                method_func, test_img, method_name
            )
            performance_metrics.append(perf_metrics)
            
            # Enhanced image
            enhanced = method_func(test_img.copy())
            
            # Image quality metrics (if reference available)
            if reference_images and i < len(reference_images):
                img_metrics = self.benchmark.calculate_image_metrics(
                    enhanced, reference_images[i]
                )
                all_metrics.append(img_metrics)
            
            # Save enhanced image
            save_path = self.output_dir / f"{method_name}_{img_name}"
            if enhanced.dtype == np.float64 or enhanced.dtype == np.float32:
                enhanced_save = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
            else:
                enhanced_save = enhanced
            
            cv2.imwrite(str(save_path), cv2.cvtColor(enhanced_save, cv2.COLOR_RGB2BGR))
        
        # Aggregate results
        result = {
            'method_name': method_name,
            'performance': {
                'avg_fps': np.mean([m['fps'] for m in performance_metrics]),
                'avg_time_ms': np.mean([m['avg_time'] for m in performance_metrics]) * 1000,
                'std_time_ms': np.mean([m['std_time'] for m in performance_metrics]) * 1000
            }
        }
        
        if all_metrics:
            result['quality'] = {
                'avg_psnr': np.mean([m['psnr'] for m in all_metrics]),
                'avg_ssim': np.mean([m['ssim'] for m in all_metrics]),
                'avg_perceptual': np.mean([m['perceptual_similarity'] for m in all_metrics])
            }
        
        self.results.append(result)
        return result
    
    def compare_methods(self, methods: List[Tuple[callable, str]], 
                       test_dir: str = "test_images"):
        """Compare multiple enhancement methods"""
        print("Starting comprehensive evaluation...")
        
        # Load test images
        test_images = self.load_test_images(test_dir)
        print(f"Loaded {len(test_images)} test images")
        
        # Evaluate each method
        for method_func, method_name in methods:
            try:
                self.evaluate_method(method_func, method_name, test_images)
            except Exception as e:
                print(f"Error evaluating {method_name}: {e}")
                continue
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        # Performance comparison
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        print(f"{'Method':<20} {'FPS':<10} {'Time(ms)':<12} {'Quality':<10}")
        print("-" * 40)
        
        for result in self.results:
            name = result['method_name']
            perf = result['performance']
            fps = perf['avg_fps']
            time_ms = perf['avg_time_ms']
            
            quality_str = "N/A"
            if 'quality' in result:
                psnr = result['quality']['avg_psnr']
                quality_str = f"{psnr:.2f} dB"
            
            print(f"{name:<20} {fps:<10.1f} {time_ms:<12.2f} {quality_str:<10}")
        
        # Save detailed results
        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: {report_file}")
        
        # Create performance visualization
        self.create_performance_plots()
    
    def create_performance_plots(self):
        """Create performance visualization plots"""
        if not self.results:
            return
        
        try:
            # Extract data for plotting
            methods = [r['method_name'] for r in self.results]
            fps_values = [r['performance']['avg_fps'] for r in self.results]
            
            # FPS comparison
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, fps_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            plt.title('Performance Comparison - Frames Per Second')
            plt.ylabel('FPS')
            plt.xlabel('Enhancement Method')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, fps in zip(bars, fps_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{fps:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Quality vs Performance scatter plot (if quality metrics available)
            quality_results = [r for r in self.results if 'quality' in r]
            if quality_results:
                plt.figure(figsize=(10, 6))
                
                fps_vals = [r['performance']['avg_fps'] for r in quality_results]
                psnr_vals = [r['quality']['avg_psnr'] for r in quality_results]
                method_names = [r['method_name'] for r in quality_results]
                
                plt.scatter(fps_vals, psnr_vals, s=100, alpha=0.7)
                
                for i, name in enumerate(method_names):
                    plt.annotate(name, (fps_vals[i], psnr_vals[i]), 
                               xytext=(5, 5), textcoords='offset points')
                
                plt.xlabel('FPS (Performance)')
                plt.ylabel('PSNR (Quality)')
                plt.title('Quality vs Performance Trade-off')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'quality_vs_performance.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Performance plots saved to: {self.output_dir}")
        
        except Exception as e:
            print(f"Could not create plots: {e}")

def create_reference_images(test_dir: str, reference_dir: str):
    """Create reference (ground truth) images from test images"""
    test_path = Path(test_dir)
    ref_path = Path(reference_dir)
    ref_path.mkdir(exist_ok=True)
    
    for img_file in test_path.glob("*.jpg"):
        img = cv2.imread(str(img_file))
        if img is not None:
            # Create a "clean" version by brightening and denoising
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Simple enhancement as reference
            img_float = img.astype(np.float32) / 255.0
            img_enhanced = np.clip(img_float * 2.5, 0, 1)  # Brighten
            img_enhanced = cv2.bilateralFilter(
                (img_enhanced * 255).astype(np.uint8), 9, 75, 75
            )
            
            save_path = ref_path / img_file.name
            cv2.imwrite(str(save_path), cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Example usage
    print("Evaluation Framework Test")
    
    # Create test framework
    evaluator = EvaluationFramework("evaluation_results")
    
    # Create sample test data
    evaluator.create_sample_test_images("test_images", 5)
    create_reference_images("test_images", "reference_images")
    
    print("Sample evaluation framework created!")
    print("This would be used to evaluate enhancement methods.")
