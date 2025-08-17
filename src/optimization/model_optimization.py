"""
Model optimization utilities for deployment.
Includes quantization, ONNX export, and Core ML conversion.
"""

import torch
import torch.quantization as quantization
import onnx
import onnxruntime as ort
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
import os


class ModelOptimizer:
    """Utilities for model optimization and deployment."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
            device: Device to run optimizations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def quantize_dynamic(self) -> torch.nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Returns:
            torch.nn.Module: Quantized model
        """
        print("Applying dynamic quantization...")
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def quantize_static(self, calibration_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Apply static quantization to the model.
        
        Args:
            calibration_loader: Data loader for calibration
            
        Returns:
            torch.nn.Module: Quantized model
        """
        print("Applying static quantization...")
        
        # Prepare model for quantization
        self.model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(self.model, inplace=True)
        
        # Calibrate with representative dataset
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= 100:  # Use only 100 batches for calibration
                    break
                
                inputs = batch['low_light'] if 'low_light' in batch else batch['input']
                _ = self.model(inputs)
        
        # Convert to quantized model
        quantized_model = quantization.convert(self.model, inplace=False)
        
        return quantized_model
    
    def export_onnx(self, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 256, 256),
                   opset_version: int = 11, optimize: bool = True) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            optimize: Whether to optimize the exported model
            
        Returns:
            str: Path to exported ONNX model
        """
        print(f"Exporting to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=optimize,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model exported successfully to {output_path}")
        return output_path
    
    def export_coreml(self, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 256, 256)) -> str:
        """
        Export model to Core ML format (macOS/iOS deployment).
        
        Args:
            output_path: Path to save Core ML model
            input_shape: Input tensor shape
            
        Returns:
            str: Path to exported Core ML model
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("coremltools not installed. Install with: pip install coremltools")
        
        print(f"Exporting to Core ML: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model.cpu(), dummy_input)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=input_shape)],
            outputs=[ct.ImageType(name="output")]
        )
        
        # Save the model
        coreml_model.save(output_path)
        
        print(f"Core ML model exported successfully to {output_path}")
        return output_path
    
    def benchmark_model(self, model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 3, 256, 256),
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            dict: Performance metrics
        """
        print(f"Benchmarking model performance...")
        
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                
                # Synchronize for accurate timing
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elif self.device == 'mps':
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
    
    def compare_models(self, models: Dict[str, torch.nn.Module], 
                      input_shape: Tuple[int, ...] = (1, 3, 256, 256)) -> Dict[str, Dict]:
        """
        Compare performance of multiple models.
        
        Args:
            models: Dictionary of model name to model mapping
            input_shape: Input tensor shape
            
        Returns:
            dict: Comparison results
        """
        results = {}
        
        for name, model in models.items():
            print(f"Benchmarking {name}...")
            
            # Move model to device
            model = model.to(self.device)
            
            # Benchmark
            performance = self.benchmark_model(model, input_shape)
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = param_count * 4 / (1024 * 1024)  # Assume float32
            
            results[name] = {
                'performance': performance,
                'parameters': param_count,
                'size_mb': model_size_mb
            }
        
        return results
    
    def optimize_for_mobile(self, output_dir: str, input_shape: Tuple[int, ...] = (1, 3, 256, 256)) -> Dict[str, str]:
        """
        Create optimized models for mobile deployment.
        
        Args:
            output_dir: Directory to save optimized models
            input_shape: Input tensor shape
            
        Returns:
            dict: Paths to optimized models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        paths = {}
        
        # 1. Quantized PyTorch model
        try:
            quantized_model = self.quantize_dynamic()
            quantized_path = os.path.join(output_dir, 'model_quantized.pth')
            torch.save(quantized_model.state_dict(), quantized_path)
            paths['quantized_pytorch'] = quantized_path
            print(f"Quantized PyTorch model saved: {quantized_path}")
        except Exception as e:
            print(f"Failed to create quantized PyTorch model: {e}")
        
        # 2. ONNX model
        try:
            onnx_path = os.path.join(output_dir, 'model.onnx')
            self.export_onnx(onnx_path, input_shape)
            paths['onnx'] = onnx_path
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")
        
        # 3. Core ML model (macOS/iOS)
        try:
            coreml_path = os.path.join(output_dir, 'model.mlmodel')
            self.export_coreml(coreml_path, input_shape)
            paths['coreml'] = coreml_path
        except Exception as e:
            print(f"Failed to export Core ML model: {e}")
        
        # 4. TorchScript model
        try:
            dummy_input = torch.randn(input_shape).to(self.device)
            scripted_model = torch.jit.trace(self.model, dummy_input)
            torchscript_path = os.path.join(output_dir, 'model_scripted.pt')
            scripted_model.save(torchscript_path)
            paths['torchscript'] = torchscript_path
            print(f"TorchScript model saved: {torchscript_path}")
        except Exception as e:
            print(f"Failed to create TorchScript model: {e}")
        
        return paths


class ONNXInference:
    """ONNX inference wrapper for optimized deployment."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to ONNX model
            providers: ONNX Runtime providers (e.g., ['CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX model loaded: {model_path}")
        print(f"Providers: {self.session.get_providers()}")
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Input tensor as numpy array
            
        Returns:
            np.ndarray: Output tensor
        """
        result = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return result[0]
    
    def benchmark(self, input_shape: Tuple[int, ...] = (1, 3, 256, 256),
                 num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark ONNX model performance.
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            
        Returns:
            dict: Performance metrics
        """
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.predict(dummy_input)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'fps': float(1.0 / np.mean(times)),
            'ms_per_frame': float(np.mean(times) * 1000)
        }


if __name__ == "__main__":
    # Example usage
    print("Model optimization utilities ready!")
    
    # This would require an actual model
    # from models import create_model
    # model = create_model('unet', 'lightweight')
    # optimizer = ModelOptimizer(model)
    # 
    # # Export models
    # paths = optimizer.optimize_for_mobile('exports/')
    # print("Exported models:", paths)
