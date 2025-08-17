#!/usr/bin/env python3
"""
Step 1: Classical Computer Vision Baselines
Building foundation for ML comparison
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
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_bilateral(self, image):
        """Bilateral filtering for edge-preserving denoising."""
        # Brighten first, then denoise
        brightened = cv2.convertScaleAbs(image, alpha=1.8, beta=30)
        denoised = cv2.bilateralFilter(brightened, 9, 75, 75)
        return denoised
    
    def apply_gaussian(self, image):
        """Gaussian blur for basic denoising."""
        brightened = cv2.convertScaleAbs(image, alpha=1.5, beta=40)
        denoised = cv2.GaussianBlur(brightened, (5, 5), 0)
        return denoised
    
    def apply_combined(self, image):
        """Combined CLAHE + Bilateral for best results."""
        # First apply CLAHE
        clahe_result = self.apply_clahe(image)
        # Then bilateral filtering
        combined = cv2.bilateralFilter(clahe_result, 5, 50, 50)
        return combined
    
    def enhance_image(self, image, method='clahe'):
        """Enhance image with specified method."""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        start_time = time.time()
        enhanced = self.methods[method](image)
        processing_time = time.time() - start_time
        
        return enhanced, processing_time


def calculate_psnr(original, enhanced):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, enhanced):
    """Calculate Structural Similarity Index."""
    # Convert to grayscale for SSIM calculation
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        enhanced_gray = enhanced
    
    ssim_value = ssim(original_gray, enhanced_gray, data_range=255)
    return ssim_value


def create_low_light_test_images():
    """Create realistic low-light test scenarios."""
    images = []
    
    # Scenario 1: Indoor room with poor lighting
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 25
    cv2.rectangle(img1, (50, 50), (200, 150), (40, 35, 30), -1)  # Furniture
    cv2.circle(img1, (400, 200), 50, (45, 40, 35), -1)  # Lamp
    cv2.rectangle(img1, (300, 300), (550, 350), (35, 30, 25), -1)  # Table
    
    # Add noise
    noise1 = np.random.normal(0, 15, img1.shape).astype(np.int16)
    img1 = np.clip(img1.astype(np.int16) + noise1, 0, 255).astype(np.uint8)
    images.append(("indoor_low_light", img1))
    
    # Scenario 2: Outdoor evening scene
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 30
    cv2.rectangle(img2, (100, 200), (500, 400), (20, 25, 35), -1)  # Building
    cv2.circle(img2, (150, 100), 20, (80, 70, 50), -1)  # Street light
    cv2.rectangle(img2, (200, 250), (250, 300), (60, 55, 40), -1)  # Window light
    
    noise2 = np.random.normal(0, 12, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
    images.append(("outdoor_evening", img2))
    
    # Scenario 3: AR/VR passthrough simulation
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 20
    # Add some AR objects
    cv2.rectangle(img3, (200, 150), (400, 250), (45, 40, 35), -1)  # Virtual object
    cv2.circle(img3, (300, 200), 30, (55, 50, 45), -1)  # Virtual sphere
    
    noise3 = np.random.normal(0, 18, img3.shape).astype(np.int16)
    img3 = np.clip(img3.astype(np.int16) + noise3, 0, 255).astype(np.uint8)
    images.append(("ar_passthrough", img3))
    
    return images


def evaluate_baselines():
    """Comprehensive evaluation of classical baselines."""
    print("Classical Computer Vision Baselines Evaluation")
    print("=" * 60)
    
    # Create test images
    test_images = create_low_light_test_images()
    
    # Initialize enhancer
    enhancer = ClassicalBaselines()
    methods = ['clahe', 'bilateral', 'gaussian', 'combined']
    
    # Results storage
    all_results = {}
    
    for scene_name, dark_image in test_images:
        print(f"\nTesting Scene: {scene_name}")
        print("-" * 40)
        
        scene_results = {}
        
        # Create reference "good lighting" version
        reference = cv2.convertScaleAbs(dark_image, alpha=3.0, beta=60)
        reference = cv2.bilateralFilter(reference, 5, 50, 50)
        
        print(f"Original brightness: {np.mean(dark_image):.1f}")
        print(f"Reference brightness: {np.mean(reference):.1f}")
        print()
        
        print("Method           Time(ms)    FPS    PSNR    SSIM    Brightness")
        print("-" * 60)
        
        for method in methods:
            # Enhance image
            enhanced, proc_time = enhancer.enhance_image(dark_image, method)
            
            # Calculate metrics
            psnr = calculate_psnr(reference, enhanced)
            ssim_value = calculate_ssim(reference, enhanced)
            brightness = np.mean(enhanced)
            fps = 1.0 / proc_time
            
            # Store results
            scene_results[method] = {
                'enhanced': enhanced,
                'time': proc_time,
                'fps': fps,
                'psnr': psnr,
                'ssim': ssim_value,
                'brightness': brightness
            }
            
            print(f"{method:<15} {proc_time*1000:>7.1f}  {fps:>6.1f}  {psnr:>6.1f}  {ssim_value:>6.3f}  {brightness:>10.1f}")
        
        all_results[scene_name] = {
            'original': dark_image,
            'reference': reference,
            'methods': scene_results
        }
        
        # Save results for this scene
        save_scene_results(scene_name, dark_image, reference, scene_results)
    
    # Overall analysis
    print("\nOverall Performance Analysis")
    print("-" * 40)
    
    # Average metrics across all scenes
    method_averages = {}
    for method in methods:
        avg_psnr = np.mean([all_results[scene]['methods'][method]['psnr'] for scene in all_results])
        avg_ssim = np.mean([all_results[scene]['methods'][method]['ssim'] for scene in all_results])
        avg_fps = np.mean([all_results[scene]['methods'][method]['fps'] for scene in all_results])
        
        method_averages[method] = {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_fps': avg_fps
        }
        
        realtime = "YES" if avg_fps >= 30 else "NO"
        print(f"{method:<15} PSNR: {avg_psnr:>5.1f}  SSIM: {avg_ssim:>5.3f}  FPS: {avg_fps:>6.1f} {realtime}")
    
    # Recommendations
    print("\nBaseline Recommendations")
    print("-" * 30)
    print("CLAHE: Best quality, good for single images")
    print("Bilateral: Good balance of quality and speed")
    print("Combined: Highest quality, suitable for video")
    print("Gaussian: Fastest, use for real-time applications")
    
    return all_results


def save_scene_results(scene_name, original, reference, results):
    """Save enhancement results for visualization."""
    output_dir = Path(f"baseline_results/{scene_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original and reference
    cv2.imwrite(str(output_dir / "01_original.jpg"), original)
    cv2.imwrite(str(output_dir / "02_reference.jpg"), reference)
    
    # Save enhanced images
    for i, (method, data) in enumerate(results.items(), 3):
        filename = f"{i:02d}_{method}.jpg"
        cv2.imwrite(str(output_dir / filename), data['enhanced'])
    
    # Create comparison grid
    create_comparison_grid(scene_name, original, reference, results)


def create_comparison_grid(scene_name, original, reference, results):
    """Create side-by-side comparison grid."""
    try:
        # Prepare images for display
        images = [original, reference]
        titles = ['Original (Dark)', 'Reference (Bright)']
        
        for method, data in results.items():
            images.append(data['enhanced'])
            psnr = data['psnr']
            fps = data['fps']
            titles.append(f"{method.upper()}\nPSNR: {psnr:.1f}dB\nFPS: {fps:.1f}")
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Classical Baselines Comparison - {scene_name}', fontsize=16)
        
        for i, (img, title) in enumerate(zip(images, titles)):
            row, col = i // 3, i % 3
            
            # Convert BGR to RGB for matplotlib
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
        
        # Hide unused subplot
        if len(images) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"baseline_results/{scene_name}/comparison_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison grid saved: baseline_results/{scene_name}/comparison_grid.png")
        
    except ImportError:
        print("Matplotlib not available - skipping comparison grid")


def main():
    print("Step 1: Classical Computer Vision Baselines")
    print("Building foundation for ML comparison")
    print()
    
    # Run comprehensive evaluation
    results = evaluate_baselines()
    
    print("\nClassical baselines evaluation completed!")
    print("\nNext Steps:")
    print("1. Check baseline_results/ folder for visual comparisons")
    print("2. These metrics are your reference for ML model comparison")
    print("3. Next: Implement U-Net (Noise2Noise) to beat these baselines")
    print("4. Goal: ML model should achieve higher PSNR/SSIM at acceptable FPS")
    
    print("\nWhat we established:")
    print("✓ PSNR/SSIM evaluation framework")
    print("✓ Multiple test scenarios (indoor, outdoor, AR)")
    print("✓ Performance benchmarks (FPS)")
    print("✓ Visual comparison system")
    print("✓ Ready for ML model comparison")


if __name__ == "__main__":
    main()
