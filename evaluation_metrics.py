import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import lpips
from skimage.metrics import structural_similarity as ssim
from model_inference import EnhancementEngine
import json
import time

class EvaluationMetrics:
    """Comprehensive evaluation metrics for image enhancement"""
    
    def __init__(self, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize LPIPS (Learned Perceptual Image Patch Similarity)
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_available = True
        except:
            print("LPIPS not available, skipping perceptual metrics")
            self.lpips_available = False
    
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float() / 255.0
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float() / 255.0
        
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index"""
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Ensure values are in [0, 1] range
        if img1.max() > 1:
            img1 = img1 / 255.0
        if img2.max() > 1:
            img2 = img2 / 255.0
        
        ssim_value = ssim(img1, img2, data_range=1.0)
        return ssim_value
    
    def calculate_lpips(self, img1, img2):
        """Calculate Learned Perceptual Image Patch Similarity"""
        if not self.lpips_available:
            return 0.0
        
        try:
            # Convert to tensors and normalize to [-1, 1]
            if isinstance(img1, np.ndarray):
                img1 = torch.from_numpy(img1).float() / 255.0
            if isinstance(img2, np.ndarray):
                img2 = torch.from_numpy(img2).float() / 255.0
            
            img1 = img1 * 2 - 1  # [0, 1] -> [-1, 1]
            img2 = img2 * 2 - 1
            
            # Ensure correct shape (B, C, H, W)
            if len(img1.shape) == 3:
                img1 = img1.unsqueeze(0)
            if len(img2.shape) == 3:
                img2 = img2.unsqueeze(0)
            
            # Permute if needed (H, W, C) -> (C, H, W)
            if img1.shape[-1] == 3:
                img1 = img1.permute(0, 3, 1, 2)
            if img2.shape[-1] == 3:
                img2 = img2.permute(0, 3, 1, 2)
            
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            with torch.no_grad():
                lpips_value = self.lpips_fn(img1, img2)
            
            return lpips_value.item()
        except Exception as e:
            print(f"LPIPS calculation error: {e}")
            return 0.0
    
    def calculate_mae(self, img1, img2):
        """Calculate Mean Absolute Error"""
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float()
        
        mae = torch.mean(torch.abs(img1 - img2))
        return mae.item()
    
    def calculate_all_metrics(self, original, enhanced):
        """Calculate all metrics"""
        metrics = {}
        
        try:
            metrics['psnr'] = self.calculate_psnr(original, enhanced)
            metrics['ssim'] = self.calculate_ssim(original, enhanced)
            metrics['lpips'] = self.calculate_lpips(original, enhanced)
            metrics['mae'] = self.calculate_mae(original, enhanced)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'mae': 0}
        
        return metrics

class EnhancementEvaluator:
    """Evaluate enhancement models on test datasets"""
    
    def __init__(self, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.metrics = EvaluationMetrics(self.device)
        self.engine = EnhancementEngine(self.device)
    
    def create_test_dataset(self, num_samples=50, img_size=256):
        """Create synthetic test dataset"""
        print("Creating test dataset...")
        
        test_dir = Path('test_data')
        test_dir.mkdir(exist_ok=True)
        
        test_images = []
        
        for i in range(num_samples):
            # Create high-quality reference image
            high_img = np.random.rand(img_size, img_size, 3) * 255
            high_img = high_img.astype(np.uint8)
            
            # Create low-light version
            brightness_factor = np.random.uniform(0.1, 0.3)
            low_img = (high_img.astype(np.float32) * brightness_factor).astype(np.uint8)
            
            # Add noise
            noise = np.random.normal(0, 10, low_img.shape)
            low_img = np.clip(low_img + noise, 0, 255).astype(np.uint8)
            
            test_images.append({
                'low': low_img,
                'high': high_img,
                'id': f'test_{i:03d}'
            })
        
        print(f"Created {num_samples} test image pairs")
        return test_images
    
    def evaluate_classical_methods(self, test_images):
        """Evaluate classical enhancement methods"""
        from step1_classical_baselines import ClassicalEnhancer
        
        enhancer = ClassicalEnhancer()
        results = {
            'clahe': [],
            'bilateral': [],
            'histogram_eq': [],
            'gaussian': []
        }
        
        print("Evaluating classical methods...")
        
        for img_data in test_images:
            low_img = img_data['low']
            high_img = img_data['high']
            
            # Test each classical method
            enhanced_clahe = enhancer.enhance_clahe(low_img)
            enhanced_bilateral = enhancer.enhance_bilateral_filter(low_img)
            enhanced_hist = enhancer.enhance_histogram_equalization(low_img)
            enhanced_gaussian = enhancer.enhance_gaussian_filter(low_img)
            
            # Calculate metrics for each method
            results['clahe'].append(self.metrics.calculate_all_metrics(high_img, enhanced_clahe))
            results['bilateral'].append(self.metrics.calculate_all_metrics(high_img, enhanced_bilateral))
            results['histogram_eq'].append(self.metrics.calculate_all_metrics(high_img, enhanced_hist))
            results['gaussian'].append(self.metrics.calculate_all_metrics(high_img, enhanced_gaussian))
        
        return results
    
    def evaluate_ai_models(self, test_images):
        """Evaluate AI enhancement models"""
        results = {
            'unet': [],
            'vit': []
        }
        
        print("Evaluating AI models...")
        
        for img_data in test_images:
            low_img = img_data['low']
            high_img = img_data['high']
            
            # Test U-Net
            enhanced_unet = self.engine.enhance_with_unet(low_img)
            results['unet'].append(self.metrics.calculate_all_metrics(high_img, enhanced_unet))
            
            # Test ViT
            enhanced_vit = self.engine.enhance_with_vit(low_img)
            results['vit'].append(self.metrics.calculate_all_metrics(high_img, enhanced_vit))
        
        return results
    
    def calculate_average_metrics(self, results):
        """Calculate average metrics across all test images"""
        averaged = {}
        
        for method, metrics_list in results.items():
            if not metrics_list:
                continue
            
            averaged[method] = {}
            metric_names = metrics_list[0].keys()
            
            for metric in metric_names:
                values = [m[metric] for m in metrics_list if not np.isnan(m[metric]) and not np.isinf(m[metric])]
                averaged[method][metric] = np.mean(values) if values else 0
        
        return averaged
    
    def benchmark_performance(self, test_image, num_iterations=100):
        """Benchmark inference speed"""
        from step1_classical_baselines import ClassicalEnhancer
        
        enhancer = ClassicalEnhancer()
        
        print("Benchmarking performance...")
        print("=" * 50)
        
        methods = {
            'CLAHE': lambda x: enhancer.enhance_clahe(x),
            'Bilateral Filter': lambda x: enhancer.enhance_bilateral_filter(x),
            'Histogram EQ': lambda x: enhancer.enhance_histogram_equalization(x),
            'Gaussian Filter': lambda x: enhancer.enhance_gaussian_filter(x),
            'U-Net': lambda x: self.engine.enhance_with_unet(x),
            'ViT': lambda x: self.engine.enhance_with_vit(x)
        }
        
        performance = {}
        
        for method_name, method_func in methods.items():
            try:
                # Warmup
                for _ in range(5):
                    _ = method_func(test_image)
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    _ = method_func(test_image)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_iterations * 1000  # ms
                fps = 1000 / avg_time
                
                performance[method_name] = {
                    'avg_time_ms': avg_time,
                    'fps': fps
                }
                
                print(f"{method_name:15}: {avg_time:6.2f} ms/frame, {fps:6.1f} FPS")
            
            except Exception as e:
                print(f"{method_name:15}: Error - {e}")
                performance[method_name] = {'avg_time_ms': 0, 'fps': 0}
        
        return performance
    
    def run_full_evaluation(self, num_test_samples=50):
        """Run complete evaluation pipeline"""
        print("Real-Time Low-Light Enhancement Evaluation")
        print("=" * 60)
        
        # Create test dataset
        test_images = self.create_test_dataset(num_test_samples)
        
        # Evaluate classical methods
        classical_results = self.evaluate_classical_methods(test_images)
        
        # Evaluate AI models
        ai_results = self.evaluate_ai_models(test_images)
        
        # Combine results
        all_results = {**classical_results, **ai_results}
        
        # Calculate averages
        averaged_results = self.calculate_average_metrics(all_results)
        
        # Performance benchmark
        test_image = test_images[0]['low']
        performance_results = self.benchmark_performance(test_image)
        
        # Print results
        self.print_evaluation_results(averaged_results, performance_results)
        
        # Save results
        final_results = {
            'quality_metrics': averaged_results,
            'performance_metrics': performance_results,
            'test_samples': num_test_samples,
            'device': str(self.device)
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nResults saved to evaluation_results.json")
        return final_results
    
    def print_evaluation_results(self, quality_results, performance_results):
        """Print formatted evaluation results"""
        print("\nQuality Metrics (Higher PSNR/SSIM, Lower LPIPS/MAE is better):")
        print("=" * 80)
        print(f"{'Method':<15} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'MAE':<8}")
        print("-" * 80)
        
        for method, metrics in quality_results.items():
            print(f"{method:<15} {metrics.get('psnr', 0):<8.2f} {metrics.get('ssim', 0):<8.4f} "
                  f"{metrics.get('lpips', 0):<8.4f} {metrics.get('mae', 0):<8.2f}")
        
        print("\nPerformance Metrics:")
        print("=" * 50)
        print(f"{'Method':<15} {'Time (ms)':<12} {'FPS':<8}")
        print("-" * 50)
        
        for method, perf in performance_results.items():
            print(f"{method:<15} {perf.get('avg_time_ms', 0):<12.2f} {perf.get('fps', 0):<8.1f}")

def main():
    """Run evaluation"""
    evaluator = EnhancementEvaluator()
    evaluator.run_full_evaluation(num_test_samples=30)

if __name__ == '__main__':
    main()
