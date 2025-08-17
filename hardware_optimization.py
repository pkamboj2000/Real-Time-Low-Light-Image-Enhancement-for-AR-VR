import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import time
import onnx
import onnxruntime as ort
from pathlib import Path
import cv2
import json
from unet_model import CompactUNet
from vit_model import EnhancementViT

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("CoreML not available. Install with: pip install coremltools")

class HardwareOptimizer:
    """Hardware optimization pipeline for AR/VR deployment"""
    
    def __init__(self, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"HardwareOptimizer initialized on {self.device}")
    
    def load_model(self, model_path, model_type='unet', img_size=256):
        """Load trained model"""
        try:
            if model_type == 'unet':
                model = CompactUNet(in_channels=3, out_channels=3, features=32)
            elif model_type == 'vit':
                model = EnhancementViT(img_size=img_size, patch_size=16, embed_dim=384, depth=6)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model file not found: {model_path}. Using random weights.")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def benchmark_model(self, model, input_size=(1, 3, 256, 256), num_runs=100):
        """Benchmark model performance"""
        print(f"Benchmarking model performance...")
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        # Memory usage estimation
        if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0  # MPS/CPU memory tracking not straightforward
        
        results = {
            'avg_time_ms': avg_time,
            'fps': fps,
            'memory_mb': memory_mb,
            'device': str(self.device)
        }
        
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Memory: {memory_mb:.1f} MB")
        
        return results
    
    def quantize_model_dynamic(self, model, output_path='quantized_model.pth'):
        """Apply dynamic quantization (int8)"""
        print(f"Applying dynamic quantization...")
        
        try:
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
            
            # Save quantized model
            torch.save(quantized_model.state_dict(), output_path)
            print(f"Quantized model saved: {output_path}")
            
            return quantized_model
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return model
    
    def export_to_onnx(self, model, output_path='model.onnx', input_size=(1, 3, 256, 256)):
        """Export model to ONNX format"""
        print(f"Exporting to ONNX...")
        
        try:
            dummy_input = torch.randn(input_size).to(self.device)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            print(f"ONNX model saved: {output_path}")
            
            # Test ONNX inference
            self.test_onnx_inference(output_path, input_size)
            
            return output_path
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            return None
    
    def test_onnx_inference(self, onnx_path, input_size=(1, 3, 256, 256)):
        """Test ONNX model inference"""
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)
            
            # Test inference
            dummy_input = np.random.randn(*input_size).astype(np.float32)
            result = session.run(None, {'input': dummy_input})
            
            print(f"  ONNX inference test passed")
            print(f"  Output shape: {result[0].shape}")
            
            # Benchmark ONNX
            start_time = time.time()
            for _ in range(50):
                _ = session.run(None, {'input': dummy_input})
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 50 * 1000
            fps = 1000 / avg_time
            
            print(f"  ONNX FPS: {fps:.1f}")
            
            return True
            
        except Exception as e:
            print(f"ONNX test failed: {e}")
            return False
    
    def export_to_coreml(self, model, output_path='model.mlmodel', input_size=(1, 3, 256, 256)):
        """Export model to Core ML format"""
        if not COREML_AVAILABLE:
            print("CoreML not available")
            return None
            
        try:
            # Trace the model
            dummy_input = torch.randn(input_size).to(self.device)
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_size)],
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.iOS15,
                outputs=[ct.TensorType(name="enhanced_image")]
            )
            
            # Save Core ML model
            coreml_model.save(output_path)
            print(f"Core ML model saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Core ML export failed: {e}")
            return None
    
    def optimize_for_deployment(self, model, model_name='enhanced_model'):
        """Complete optimization pipeline"""
        print(f"\nStarting optimization for {model_name}")
        
        results = {}
        
        # 1. Benchmark original model
        print("\n1. Benchmarking original model...")
        original_results = self.benchmark_model(model)
        results['original'] = original_results
        
        # 2. Apply quantization
        print("\n2. Applying quantization...")
        quantized_model = self.quantize_model_dynamic(model, f'{model_name}_quantized.pth')
        if quantized_model is not None:
            quantized_results = self.benchmark_model(quantized_model)
            results['quantized'] = quantized_results
        
        # 3. Export to ONNX
        print("\n3. Exporting to ONNX...")
        onnx_path = self.export_to_onnx(model, f'{model_name}.onnx')
        if onnx_path:
            results['onnx_export'] = True
        
        # 4. Export to Core ML
        print("\n4. Exporting to Core ML...")
        coreml_path = self.export_to_coreml(model, f'{model_name}.mlmodel')
        if coreml_path:
            results['coreml_export'] = True
        
        # Summary
        print("\nOptimization Summary:")
        print(f"  Original FPS: {original_results['fps']:.1f}")
        if 'quantized' in results:
            print(f"  Quantized FPS: {results['quantized']['fps']:.1f}")
        print(f"  ONNX Export: {'Success' if results.get('onnx_export') else 'Failed'}")
        print(f"  Core ML Export: {'Success' if results.get('coreml_export') else 'Failed'}")
        
        return results

def create_demo_models():
    """Create sample models for testing optimization"""
    # Create a simple U-Net
    unet = CompactUNet(in_channels=3, out_channels=3, features=32)
    
    # Create a simple ViT
    vit = EnhancementViT(img_size=256, patch_size=16, embed_dim=384, depth=6)
    
    # Save models
    torch.save(unet.state_dict(), 'demo_unet.pth')
    torch.save(vit.state_dict(), 'demo_vit.pth')
    
    return unet, vit

def main():
    """Demo of hardware optimization pipeline"""
    print("Demo models created:")
    unet, vit = create_demo_models()
    
    optimizer = HardwareOptimizer()
    
    print("Hardware Optimization Pipeline Demo")
    print("=" * 50)
    
    # Test U-Net optimization
    print("\nOptimizing U-Net model...")
    unet_results = optimizer.optimize_for_deployment(unet, 'unet_optimized')
    
    # Test ViT optimization  
    print("\nOptimizing ViT model...")
    vit_results = optimizer.optimize_for_deployment(vit, 'vit_optimized')
    
    print("\nOptimization complete!")

if __name__ == "__main__":
    main()
