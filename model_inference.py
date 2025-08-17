import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from compact_unet import CompactUNet
from enhancement_vit import EnhancementViT
import torchvision.transforms as transforms
from PIL import Image
import time

class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.models = {}
    
    def load_unet(self, model_path='best_unet_model.pth', img_size=256):
        """Load U-Net model"""
        try:
            model = CompactUNet(in_channels=3, out_channels=3, features=32)
            
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded U-Net from {model_path}")
            else:
                print(f"Model file {model_path} not found. Using random weights.")
            
            model.to(self.device)
            model.eval()
            self.models['unet'] = model
            return model
        except Exception as e:
            print(f"Error loading U-Net: {e}")
            return None
    
    def load_vit(self, model_path='best_vit_model.pth', img_size=256):
        """Load ViT model"""
        try:
            model = EnhancementViT(img_size=img_size, patch_size=16, embed_dim=384, depth=6)
            
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded ViT from {model_path}")
            else:
                print(f"Model file {model_path} not found. Using random weights.")
            
            model.to(self.device)
            model.eval()
            self.models['vit'] = model
            return model
        except Exception as e:
            print(f"Error loading ViT: {e}")
            return None

class EnhancementEngine:
    """Real-time enhancement engine"""
    
    def __init__(self, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.loader = ModelLoader(self.device)
        self.models = {}
        
        # Load models
        self.models['unet'] = self.loader.load_unet()
        self.models['vit'] = self.loader.load_vit()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.inverse_transform = transforms.ToPILImage()
    
    def preprocess_image(self, image, target_size=256):
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Store original size
        original_size = image.size
        
        # Resize and convert to tensor
        image = image.resize((target_size, target_size))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    def postprocess_output(self, output_tensor, original_size):
        """Convert model output back to image"""
        # Move to CPU and remove batch dimension
        output = output_tensor.cpu().squeeze(0)
        
        # Convert to PIL image
        image = self.inverse_transform(output)
        
        # Resize back to original size
        image = image.resize(original_size)
        
        # Convert to numpy array
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def enhance_with_unet(self, image):
        """Enhance image using U-Net"""
        if self.models['unet'] is None:
            return image
        
        try:
            input_tensor, original_size = self.preprocess_image(image)
            
            with torch.no_grad():
                output = self.models['unet'](input_tensor)
            
            enhanced = self.postprocess_output(output, original_size)
            return enhanced
        except Exception as e:
            print(f"U-Net enhancement error: {e}")
            return image
    
    def enhance_with_vit(self, image):
        """Enhance image using ViT"""
        if self.models['vit'] is None:
            return image
        
        try:
            input_tensor, original_size = self.preprocess_image(image)
            
            with torch.no_grad():
                output = self.models['vit'](input_tensor)
            
            enhanced = self.postprocess_output(output, original_size)
            return enhanced
        except Exception as e:
            print(f"ViT enhancement error: {e}")
            return image
    
    def benchmark_models(self, image, num_iterations=100):
        """Benchmark model performance"""
        print("Benchmarking AI Models...")
        print("=" * 40)
        
        # Preprocess once
        input_tensor, original_size = self.preprocess_image(image)
        
        # Benchmark U-Net
        if self.models['unet'] is not None:
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.models['unet'](input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            fps = 1000 / avg_time
            print(f"U-Net: {avg_time:.2f} ms/frame, {fps:.1f} FPS")
        
        # Benchmark ViT
        if self.models['vit'] is not None:
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.models['vit'](input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            fps = 1000 / avg_time
            print(f"ViT: {avg_time:.2f} ms/frame, {fps:.1f} FPS")

def demo_ai_models():
    """Demo AI model inference"""
    print("AI Model Inference Demo")
    print("=" * 30)
    
    # Initialize enhancement engine
    engine = EnhancementEngine()
    
    # Create test image
    test_image = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)  # Dark image
    
    # Test U-Net enhancement
    print("Testing U-Net enhancement...")
    enhanced_unet = engine.enhance_with_unet(test_image)
    print(f"U-Net output shape: {enhanced_unet.shape}")
    
    # Test ViT enhancement
    print("Testing ViT enhancement...")
    enhanced_vit = engine.enhance_with_vit(test_image)
    print(f"ViT output shape: {enhanced_vit.shape}")
    
    # Benchmark performance
    engine.benchmark_models(test_image, num_iterations=50)
    
    print("\nAI models are ready for real-time inference!")

if __name__ == '__main__':
    demo_ai_models()
