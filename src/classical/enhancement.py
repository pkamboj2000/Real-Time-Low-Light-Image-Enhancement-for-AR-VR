"""
Classical image enhancement methods for low-light conditions.

Traditional computer vision approaches for image enhancement that don't
require training. These methods provide good baseline performance and 
can run on any hardware.

Author: Pranjal Kamboj
Created: August 2025
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ClassicalEnhancer:
    """
    Collection of classical image enhancement techniques for low-light conditions.
    
    Implements 8 different enhancement methods including histogram equalization,
    CLAHE, gamma correction, and various filtering approaches. These methods
    provide good performance without requiring GPU or model training.
    """

import cv2
import numpy as np
from typing import Optional, Tuple
import time


class ClassicalEnhancer:
    """Classical methods for low-light image enhancement."""
    
    def __init__(self):
        self.methods = {
            'histogram_eq': self.histogram_equalization,
            'clahe': self.clahe_enhancement,
            'gamma_correction': self.gamma_correction,
            'log_transform': self.log_transform,
            'unsharp_mask': self.unsharp_masking,
            'bilateral_filter': self.bilateral_filtering,
            'gaussian_filter': self.gaussian_filtering,
            'combined': self.combined_enhancement
        }
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return enhanced
    
    def clahe_enhancement(self, image: np.ndarray, clip_limit: float = 2.0, 
                         tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image (np.ndarray): Input image in BGR format
            clip_limit (float): Threshold for contrast limiting
            tile_grid_size (tuple): Size of grid for histogram equalization
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        """
        Apply gamma correction for brightness adjustment.
        
        Args:
            image (np.ndarray): Input image
            gamma (float): Gamma value (< 1 brightens, > 1 darkens)
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        corrected = np.power(normalized, gamma)
        
        # Convert back to uint8
        enhanced = (corrected * 255).astype(np.uint8)
        return enhanced
    
    def log_transform(self, image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Apply logarithmic transformation to enhance dark regions.
        
        Args:
            image (np.ndarray): Input image
            c (float): Scaling constant
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply log transform
        log_transformed = c * np.log(1 + normalized)
        
        # Normalize to [0, 255]
        enhanced = ((log_transformed / log_transformed.max()) * 255).astype(np.uint8)
        return enhanced
    
    def unsharp_masking(self, image: np.ndarray, kernel_size: int = 5, 
                       sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.
        
        Args:
            image (np.ndarray): Input image
            kernel_size (int): Size of Gaussian kernel
            sigma (float): Standard deviation for Gaussian kernel
            amount (float): Strength of the sharpening
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Calculate unsharp mask
        mask = cv2.subtract(image, blurred)
        
        # Apply mask
        enhanced = cv2.addWeighted(image, 1, mask, amount, 0)
        return enhanced
    
    def bilateral_filtering(self, image: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filtering for noise reduction while preserving edges.
        
        Args:
            image (np.ndarray): Input image
            d (int): Diameter of pixel neighborhood
            sigma_color (float): Filter sigma in color space
            sigma_space (float): Filter sigma in coordinate space
            
        Returns:
            np.ndarray: Enhanced image
        """
        enhanced = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return enhanced
    
    def gaussian_filtering(self, image: np.ndarray, kernel_size: int = 5, 
                          sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian filtering for noise reduction.
        
        Args:
            image (np.ndarray): Input image
            kernel_size (int): Size of Gaussian kernel
            sigma (float): Standard deviation for Gaussian kernel
            
        Returns:
            np.ndarray: Enhanced image
        """
        enhanced = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return enhanced
    
    def combined_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a combination of enhancement techniques.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Step 1: Gamma correction for brightness
        enhanced = self.gamma_correction(image, gamma=0.6)
        
        # Step 2: CLAHE for contrast
        enhanced = self.clahe_enhancement(enhanced, clip_limit=3.0)
        
        # Step 3: Bilateral filtering for noise reduction
        enhanced = self.bilateral_filtering(enhanced, d=5, sigma_color=50, sigma_space=50)
        
        # Step 4: Mild unsharp masking for sharpness
        enhanced = self.unsharp_masking(enhanced, amount=0.5)
        
        return enhanced
    
    def enhance_image(self, image: np.ndarray, method: str = 'combined', 
                     **kwargs) -> Tuple[np.ndarray, float]:
        """
        Enhance image using specified method.
        
        Args:
            image (np.ndarray): Input image
            method (str): Enhancement method to use
            **kwargs: Additional parameters for the method
            
        Returns:
            tuple: (enhanced_image, processing_time)
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(self.methods.keys())}")
        
        start_time = time.time()
        enhanced = self.methods[method](image, **kwargs)
        processing_time = time.time() - start_time
        
        return enhanced, processing_time
    
    def get_method_info(self) -> dict:
        """
        Get information about available enhancement methods.
        
        Returns:
            dict: Method information
        """
        return {
            'histogram_eq': 'Global histogram equalization',
            'clahe': 'Contrast Limited Adaptive Histogram Equalization',
            'gamma_correction': 'Gamma correction for brightness adjustment',
            'log_transform': 'Logarithmic transformation',
            'unsharp_mask': 'Unsharp masking for edge enhancement',
            'bilateral_filter': 'Bilateral filtering for noise reduction',
            'gaussian_filter': 'Gaussian filtering for smoothing',
            'combined': 'Combination of multiple techniques'
        }


def enhance_image_classical(image_path: str, output_path: Optional[str] = None, 
                          method: str = 'combined', **kwargs) -> np.ndarray:
    """
    Convenience function to enhance an image using classical methods.
    
    Args:
        image_path (str): Path to input image
        output_path (str, optional): Path to save enhanced image
        method (str): Enhancement method to use
        **kwargs: Additional parameters for the method
        
    Returns:
        np.ndarray: Enhanced image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    # Create enhancer
    enhancer = ClassicalEnhancer()
    
    # Enhance image
    enhanced, processing_time = enhancer.enhance_image(image, method, **kwargs)
    
    print(f"Enhancement completed in {processing_time:.3f} seconds using {method}")
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, enhanced)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classical Image Enhancement")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--method", type=str, default="combined", 
                       help="Enhancement method to use")
    
    args = parser.parse_args()
    
    try:
        enhanced = enhance_image_classical(args.input, args.output, args.method)
        print("Enhancement completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
