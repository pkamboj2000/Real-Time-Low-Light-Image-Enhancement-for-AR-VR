#!/usr/bin/env python3
"""
 Model Optimization and Export Tools for AR/VR Enhancement
Quantization, ONNX export, Core ML conversion for deployment
"""

import torch
import torch.nn as nn
import numpy as np
import time
import onnx
import json
from pathlib import Path


class CompactUNet(nn.Module):
    """Same U-Net architecture for optimization."""
    
    def __init__(self, in_channels=3, out_channels=3, base_filters=16):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        
        # Output
        self.final = nn.Conv2d(base_filters, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        output = torch.sigmoid(self.final(d1))
        return output


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def measure_model_performance(model, device, input_shape=(1, 3, 256, 256), num_runs=50):
    """Measure model inference performance."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Actual timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(dummy_input)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    return avg_time, fps, times


def quantize_model_dynamic(model):
    """Apply dynamic quantization (weights only)."""
    print(" Applying dynamic quantization...")
    
    # Dynamic quantization - works on CPU
    quantized_model = torch.quantization.quantize_dynamic(
        model.cpu(), 
        {nn.Conv2d, nn.ConvTranspose2d}, 
        dtype=torch.qint8
    )
    
    # Get model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
    
    print(f" Dynamic quantization complete:")
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   Quantized size: {quantized_size:.2f} MB")
    print(f"   Compression ratio: {original_size/quantized_size:.2f}x")
    
    return quantized_model


def export_to_onnx(model, output_path="models/unet_arvr_enhancer.onnx", input_shape=(1, 3, 256, 256)):
    """Export model to ONNX format."""
    print(" Exporting to ONNX format...")
    
    model.eval()
    model = model.cpu()  # ONNX export works better on CPU
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
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
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f" ONNX export successful:")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Opset version: 11")
        
        return True
    except Exception as e:
        print(f" ONNX export failed: {e}")
        return False


def export_to_coreml(model, output_path="models/unet_arvr_enhancer.mlpackage", input_shape=(1, 3, 256, 256)):
    """Export model to Core ML format."""
    try:
        import coremltools as ct
        print(" Exporting to Core ML format...")
        
        model.eval()
        model = model.cpu()
        
        # Create example input
        example_input = torch.randn(input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=input_shape)],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.ALL  # Use Neural Engine when available
        )
        
        # Add metadata
        coreml_model.short_description = "Real-Time Low-Light Enhancement for AR/VR"
        coreml_model.author = "AR/VR Enhancement System"
        coreml_model.license = "MIT"
        coreml_model.version = "1.0"
        
        # Save model
        coreml_model.save(output_path)
        
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f" Core ML export successful:")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Compute units: ALL (Neural Engine + GPU + CPU)")
        
        return True
        
    except ImportError:
        print("⚠️ Core ML Tools not available. Install with: pip install coremltools")
        return False
    except Exception as e:
        print(f" Core ML export failed: {e}")
        return False


def create_fp16_model(model, device):
    """Create FP16 version of the model."""
    if device.type == 'cuda':
        print(" Creating FP16 model for CUDA...")
        fp16_model = model.half().to(device)
        return fp16_model
    else:
        print("⚠️ FP16 not supported on current device, returning original model")
        return model


def benchmark_all_variants(original_model, device):
    """Benchmark all model variants."""
    print("\n🏁 Comprehensive Model Benchmarking")
    print("=" * 60)
    
    results = {}
    
    # Original model
    print("\n Original Model (FP32)")
    avg_time, fps, _ = measure_model_performance(original_model, device)
    original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 * 1024)
    
    results['original_fp32'] = {
        'avg_time_ms': avg_time * 1000,
        'fps': fps,
        'size_mb': original_size,
        'realtime_capable': fps >= 30
    }
    
    print(f"   Inference time: {avg_time*1000:.2f}ms")
    print(f"   FPS: {fps:.1f}")
    print(f"   Size: {original_size:.2f} MB")
    print(f"   Real-time: {'' if fps >= 30 else ''}")
    
    # FP16 model (if supported)
    if device.type == 'cuda':
        print("\n FP16 Model")
        fp16_model = create_fp16_model(original_model.clone(), device)
        avg_time_fp16, fps_fp16, _ = measure_model_performance(fp16_model, device)
        fp16_size = sum(p.numel() * p.element_size() for p in fp16_model.parameters()) / (1024 * 1024)
        
        results['fp16'] = {
            'avg_time_ms': avg_time_fp16 * 1000,
            'fps': fps_fp16,
            'size_mb': fp16_size,
            'realtime_capable': fps_fp16 >= 30
        }
        
        print(f"   Inference time: {avg_time_fp16*1000:.2f}ms ({fps_fp16/fps:.2f}x speedup)")
        print(f"   FPS: {fps_fp16:.1f}")
        print(f"   Size: {fp16_size:.2f} MB ({original_size/fp16_size:.1f}x smaller)")
        print(f"   Real-time: {'' if fps_fp16 >= 30 else ''}")
    
    # Quantized model
    print("\n Quantized Model (INT8)")
    import copy
    quantized_model = quantize_model_dynamic(copy.deepcopy(original_model))
    avg_time_quant, fps_quant, _ = measure_model_performance(quantized_model, torch.device('cpu'))
    quant_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
    
    results['quantized_int8'] = {
        'avg_time_ms': avg_time_quant * 1000,
        'fps': fps_quant,
        'size_mb': quant_size,
        'realtime_capable': fps_quant >= 30
    }
    
    print(f"   Inference time: {avg_time_quant*1000:.2f}ms")
    print(f"   FPS: {fps_quant:.1f}")
    print(f"   Size: {quant_size:.2f} MB ({original_size/quant_size:.1f}x smaller)")
    print(f"   Real-time: {'' if fps_quant >= 30 else ''}")
    
    return results


def main():
    """Main optimization and export pipeline."""
    print(" AR/VR Enhancement Model Optimization & Export")
    print("=" * 60)
    print(" Goal: Optimize models for real-time AR/VR deployment")
    print()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print("\n Loading trained model...")
    model = CompactUNet()
    
    try:
        model.load_state_dict(torch.load('models/compact_unet_arvr.pth', map_location='cpu'))
        print(" Trained model loaded successfully")
        model_trained = True
    except FileNotFoundError:
        print("⚠️ Trained model not found, using random weights for demo")
        model_trained = False
    
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("optimized_models").mkdir(exist_ok=True)
    
    # Benchmark all variants
    benchmark_results = benchmark_all_variants(model, device)
    
    # Export to different formats
    print("\n Exporting Models")
    print("-" * 30)
    
    # ONNX export
    onnx_success = export_to_onnx(
        model.cpu(), 
        "optimized_models/unet_arvr_enhancer.onnx"
    )
    
    # Core ML export
    coreml_success = export_to_coreml(
        model.cpu(), 
        "optimized_models/unet_arvr_enhancer.mlpackage"
    )
    
    # Quantized model export
    print("\n Saving Quantized Model...")
    quantized_model = quantize_model_dynamic(model.cpu())
    torch.save(quantized_model.state_dict(), 'optimized_models/unet_arvr_quantized.pth')
    print(" Quantized model saved")
    
    # Create deployment package
    print("\n Creating Deployment Package...")
    
    deployment_info = {
        "model_info": {
            "architecture": "Compact U-Net",
            "parameters": param_count,
            "input_shape": [1, 3, 256, 256],
            "output_shape": [1, 3, 256, 256],
            "trained": model_trained
        },
        "performance_benchmarks": {
            k: {
                'avg_time_ms': float(v['avg_time_ms']),
                'fps': float(v['fps']),
                'realtime_capable': bool(v['realtime_capable'])
            } for k, v in benchmark_results.items()
        },
        "available_formats": {
            "pytorch_fp32": "models/compact_unet_arvr.pth",
            "pytorch_quantized": "optimized_models/unet_arvr_quantized.pth",
            "onnx": "optimized_models/unet_arvr_enhancer.onnx" if onnx_success else None,
            "coreml": "optimized_models/unet_arvr_enhancer.mlpackage" if coreml_success else None
        },
        "deployment_recommendations": {
            "ios_macos": "Core ML model for Neural Engine acceleration",
            "android": "ONNX model with TensorFlow Lite conversion",
            "web": "ONNX.js with ONNX model",
            "embedded": "Quantized PyTorch model",
            "server": "Original FP32 or FP16 model"
        },
        "real_time_capability": {
            format_name: bool(data['realtime_capable']) 
            for format_name, data in benchmark_results.items()
        }
    }
    
    with open('optimized_models/deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(" Deployment package created")
    
    # Summary
    print("\n Optimization Summary")
    print("-" * 40)
    print(" Model quantization completed (INT8)")
    if onnx_success:
        print(" ONNX export successful")
    if coreml_success:
        print(" Core ML export successful")
    print(" Performance benchmarks completed")
    print(" Deployment package ready")
    
    print("\n📁 Generated Files:")
    print(" optimized_models/unet_arvr_enhancer.onnx")
    print(" optimized_models/unet_arvr_enhancer.mlpackage")
    print(" optimized_models/unet_arvr_quantized.pth")
    print(" optimized_models/deployment_info.json")
    
    print("\n Deployment Recommendations:")
    print(" iOS/macOS: Use Core ML model for Neural Engine")
    print(" Android: Convert ONNX to TensorFlow Lite")
    print(" Web: Use ONNX.js with ONNX model")
    print(" Edge devices: Use quantized PyTorch model")
    
    # Best performing model
    best_model = max(benchmark_results.items(), key=lambda x: x[1]['fps'])
    print(f"\n Best Performance: {best_model[0]} ({best_model[1]['fps']:.1f} FPS)")
    
    return benchmark_results


if __name__ == "__main__":
    try:
        results = main()
        print("\n Model optimization and export completed successfully!")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
