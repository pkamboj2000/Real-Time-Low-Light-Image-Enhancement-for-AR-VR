"""
Demo script for classical enhancement methods.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from enhancement import ClassicalEnhancer


def create_comparison_plot(original, enhanced, method_name, save_path=None):
    """Create side-by-side comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(enhanced_rgb)
    axes[1].set_title(f'Enhanced ({method_name})', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def demo_all_methods(image_path, output_dir=None):
    """Demonstrate all classical enhancement methods."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    # Create enhancer
    enhancer = ClassicalEnhancer()
    
    # Get all methods
    methods = enhancer.get_method_info()
    
    print(f"Loaded image: {image.shape}")
    print(f"Testing {len(methods)} enhancement methods...\n")
    
    results = {}
    
    for method_name, description in methods.items():
        print(f"Testing {method_name}: {description}")
        
        try:
            enhanced, processing_time = enhancer.enhance_image(image, method_name)
            results[method_name] = {
                'enhanced': enhanced,
                'time': processing_time,
                'description': description
            }
            
            print(f"  ✓ Completed in {processing_time:.3f}s")
            
            # Save enhanced image if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{method_name}_enhanced.jpg")
                cv2.imwrite(output_path, enhanced)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Create summary comparison
    if output_dir:
        create_summary_comparison(image, results, output_dir)
    
    return results


def create_summary_comparison(original, results, output_dir):
    """Create a summary comparison of all methods."""
    methods = list(results.keys())[:8]  # Limit to 8 methods for display
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Original image
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Enhanced images
    for i, method in enumerate(methods):
        if i + 1 < len(axes):
            enhanced_rgb = cv2.cvtColor(results[method]['enhanced'], cv2.COLOR_BGR2RGB)
            axes[i + 1].imshow(enhanced_rgb)
            time_str = f"{results[method]['time']:.3f}s"
            axes[i + 1].set_title(f'{method}\n({time_str})', fontsize=10)
            axes[i + 1].axis('off')
    
    # Hide unused axes
    for i in range(len(methods) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'enhancement_comparison.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary comparison saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Classical Enhancement Demo")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input image path")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory for results")
    parser.add_argument("--method", type=str, 
                       help="Specific method to test (if not provided, tests all)")
    
    args = parser.parse_args()
    
    try:
        if args.method:
            # Test specific method
            enhancer = ClassicalEnhancer()
            image = cv2.imread(args.input)
            enhanced, processing_time = enhancer.enhance_image(image, args.method)
            
            print(f"Enhanced using {args.method} in {processing_time:.3f}s")
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f"{args.method}_enhanced.jpg")
                cv2.imwrite(output_path, enhanced)
                
                # Create comparison plot
                comparison_path = os.path.join(args.output_dir, f"{args.method}_comparison.png")
                create_comparison_plot(image, enhanced, args.method, comparison_path)
        else:
            # Test all methods
            results = demo_all_methods(args.input, args.output_dir)
            
            # Print summary
            print("=" * 50)
            print("ENHANCEMENT SUMMARY")
            print("=" * 50)
            for method, result in results.items():
                print(f"{method:20} | {result['time']:6.3f}s | {result['description']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
