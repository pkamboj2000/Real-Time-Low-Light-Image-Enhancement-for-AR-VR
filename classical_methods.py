#!/usr/bin/env python3
"""
Classical Computer Vision Baselines
Foundation methods for low-light enhancement
"""

import cv2
import numpy as np
import time
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class ClassicalBaselines:
    """Classical computer vision methods for low-light enhancement."""
    
    def __init__(self):
        self.methods = {
            'clahe': self.apply_clahe,
            'bilateral': self.apply_bilateral,
            'gaussian': self.apply_gaussian,
            'combined': self.apply_combined
        }
    
    def apply_clahe(self, image):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_bilateral(self, image):
        """Bilateral filtering for noise reduction while preserving edges."""
        # Apply bilateral filter multiple times for better enhancement
        result = image.copy()
        for _ in range(2):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        # Enhance brightness
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=30)
        
        return result
    
    def apply_gaussian(self, image):
        """Gaussian-based enhancement with unsharp masking."""
        # Create gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Unsharp masking
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Brightness enhancement
        result = cv2.convertScaleAbs(unsharp, alpha=1.1, beta=25)
        
        return result
    
    def apply_combined(self, image):
        """Combined approach using multiple classical methods."""
        # Step 1: CLAHE for contrast
        clahe_result = self.apply_clahe(image)
        
        # Step 2: Bilateral filtering for noise reduction
        bilateral_result = cv2.bilateralFilter(clahe_result, 7, 50, 50)
        
        # Step 3: Gamma correction
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(bilateral_result, table)
        
        return result
    
    def apply_histogram_equalization(self, image):
        """Standard histogram equalization."""
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to BGR
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return result
    
    def enhance_image(self, image, method='combined'):
        """Apply selected enhancement method."""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        start_time = time.time()
        result = self.methods[method](image)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return result, processing_time
    
    def benchmark_methods(self, image, save_results=False):
        """Benchmark all classical methods."""
        results = {}
        
        print("Benchmarking classical methods...")
        
        for method_name, method_func in self.methods.items():
            # Time the method
            start_time = time.time()
            enhanced = method_func(image)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            fps = 1000 / processing_time if processing_time > 0 else float('inf')
            
            results[method_name] = {
                'image': enhanced,
                'time_ms': processing_time,
                'fps': fps
            }
            
            print(f"  {method_name}: {processing_time:.2f} ms ({fps:.1f} FPS)")
        
        if save_results:
            self.save_comparison(image, results)
        
        return results
    
    def save_comparison(self, original, results):
        """Save comparison of all methods."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Enhanced images
        for i, (method, result) in enumerate(results.items(), 1):
            if i < len(axes):
                axes[i].imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
                axes[i].set_title(f'{method.title()}\n{result["time_ms"]:.2f}ms')
                axes[i].axis('off')
        
        # Hide unused subplot
        if len(results) + 1 < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('classical_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison saved as 'classical_comparison.png'")
    
    def calculate_metrics(self, original, enhanced):
        """Calculate quality metrics."""
        # Convert to grayscale for SSIM
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # SSIM
        ssim_value = ssim(orig_gray, enh_gray)
        
        # MSE and PSNR
        mse = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Brightness improvement
        orig_brightness = np.mean(original)
        enh_brightness = np.mean(enhanced)
        brightness_improvement = (enh_brightness - orig_brightness) / orig_brightness * 100
        
        return {
            'ssim': ssim_value,
            'psnr': psnr,
            'brightness_improvement': brightness_improvement
        }


def create_test_image():
    """Create a synthetic low-light test image."""
    # Create a simple scene
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add some geometric shapes with low brightness
    cv2.rectangle(image, (50, 50), (200, 150), (40, 40, 40), -1)
    cv2.circle(image, (400, 200), 80, (60, 60, 60), -1)
    cv2.rectangle(image, (300, 300), (550, 350), (30, 30, 30), -1)
    
    # Add noise to simulate low-light conditions
    noise = np.random.normal(0, 10, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return noisy_image


def load_test_images():
    """Load test images from common formats."""
    test_images = []
    
    # Try to load from common test image paths
    test_paths = [
        'test_image.jpg',
        'test_image.png', 
        'sample.jpg',
        'sample.png'
    ]
    
    for path in test_paths:
        if Path(path).exists():
            img = cv2.imread(str(path))
            if img is not None:
                test_images.append(img)
                print(f"Loaded test image: {path}")
    
    # If no images found, create synthetic one
    if not test_images:
        print("No test images found, creating synthetic image")
        test_images.append(create_test_image())
    
    return test_images


def main():
    """Main demo function for classical methods."""
    print("Classical Low-Light Enhancement Demo")
    print("=" * 40)
    
    # Initialize classical methods
    classical = ClassicalBaselines()
    
    # Load or create test images
    test_images = load_test_images()
    
    for i, image in enumerate(test_images):
        print(f"\nProcessing image {i+1}...")
        
        # Benchmark all methods
        results = classical.benchmark_methods(image, save_results=True)
        
        # Calculate metrics for the combined method
        combined_result = results['combined']['image']
        metrics = classical.calculate_metrics(image, combined_result)
        
        print("\nQuality metrics (Combined method):")
        print(f"  SSIM: {metrics['ssim']:.3f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Brightness improvement: {metrics['brightness_improvement']:.1f}%")
    
    print("\nClassical methods evaluation complete!")


if __name__ == "__main__":
    main()
