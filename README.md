# Real-Time Low-Light Image Enhancement for AR/VR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Real-Time Low-Light Image Enhancement for AR/VR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive solution for real-time low-light image enhancement specifically designed for AR/VR applications. This project combines classical computer vision techniques with modern deep learning approaches to improve image quality in challenging lighting conditions.

## Project Overview

Low-light image quality is a persistent challenge in AR/VR devices, especially in indoor environments. This project addresses this issue by implementing multiple enhancement approaches that can run in real-time on consumer hardware.

**Key Objectives:**
- Real-time processing for AR/VR applications
- Multiple enhancement algorithms (classical + deep learning)
- Hardware optimization for mobile devices
- Comprehensive quality evaluation metrics

### Features

- **Classical Enhancement Methods**: Histogram equalization, CLAHE, gamma correction, bilateral filtering
- **Deep Learning Models**: U-Net and Vision Transformer architectures optimized for low-light enhancement  
- **Hardware Acceleration**: Support for Apple Silicon (MPS), CUDA, and optimized CPU inference
- **Model Export**: ONNX and Core ML format support for mobile deployment
- **Quality Metrics**: PSNR, SSIM, LPIPS for comprehensive evaluation
- **Interactive Interface**: Streamlit dashboard for real-time testing and parameter tuning
- **AR/VR Dataset Support**: Optimized for egocentric camera data

## Getting Started

### Installation

```bash
# Quick setup using the provided script
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate

# Verify installation
python test_setup.py
```

### Basic Usage

```bash
# Test classical enhancement methods
python src/classical/demo.py --input your_image.jpg --output_dir results/

# Launch the interactive dashboard
streamlit run dashboard/app.py

This is my attempt at creating real-time low-light image enhancement specifically for AR/VR applications. I've combined classical computer vision methods with some modern deep learning approaches, trying to keep things efficient enough to actually run on real hardware.

## 🎯 What's This About?

Ever tried using an AR headset in a dimly lit room? Yeah, the camera feed looks terrible. I got frustrated with this while working on some AR projects and decided to do something about it. 

The goal is simple: make low-light images from AR/VR headsets look way better, in real-time, without killing the battery. Easier said than done, but hey, that's what makes it fun!

### What I've Built

- 🔧 **Classical Methods**: Started with the basics - histogram equalization, CLAHE, some denoising. Old but gold!
- 🤖 **Deep Learning**: U-Net and Vision Transformer implementations. The U-Net works surprisingly well for this
- ⚡ **Apple Silicon Optimization**: Finally put my M2 MacBook to good use! MPS backend is pretty sweet
- 📱 **Mobile Export**: ONNX and Core ML exports because nobody wants a model that only runs on desktop
- 📊 **Proper Evaluation**: PSNR, SSIM, LPIPS - gotta measure what matters
- 🎮 **Interactive Demo**: Streamlit dashboard because showing beats telling every time
- 🎯 **AR/VR Focus**: Tested specifically with egocentric video data (thanks Meta for the AEA dataset!)

## � Quick Start

### Installation

```bash
# Run automated setup
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Test installation
python test_setup.py
```

### Try It Out

```bash
# Test classical enhancement (no training required)
python src/classical/demo.py --input path/to/low_light_image.jpg --output_dir results/

# Launch interactive dashboard
streamlit run dashboard/app.py
```

## �📁 Project Structure

```
├── data/                    # Dataset storage
│   ├── raw/                # Original datasets (LOL, SID, AEA)
│   ├── processed/          # Preprocessed data
│   └── samples/            # Sample images for testing
├── src/                    # Source code
│   ├── models/             # Deep learning models (U-Net, ViT)
│   ├── classical/          # Classical enhancement methods
│   ├── data/               # Data loading and preprocessing
│   ├── training/           # Training scripts and utilities
│   ├── evaluation/         # Evaluation metrics (PSNR, SSIM, LPIPS)
│   ├── optimization/       # Hardware optimization and export
│   └── utils/              # Utility functions
├── dashboard/              # Interactive Streamlit interface
├── notebooks/              # Jupyter notebooks for experiments
├── configs/                # Training configuration files
├── models/                 # Saved model checkpoints
├── exports/                # Exported models (ONNX, Core ML)
└── results/                # Evaluation results and comparisons
```

## 🛠️ Supported Methods

### Classical Enhancement
- **Histogram Equalization**: Global contrast enhancement
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Gamma Correction**: Brightness adjustment
- **Bilateral Filtering**: Edge-preserving noise reduction
- **Unsharp Masking**: Sharpness enhancement
- **Combined Method**: Optimized pipeline of multiple techniques

### Deep Learning Models
- **U-Net**: Standard and lightweight variants for semantic enhancement
- **Vision Transformer**: Patch-based processing with attention mechanisms
- **Self-Supervised Training**: Noise2Noise approach for unpaired learning

### Hardware Optimization
- **Apple Silicon**: Native MPS backend support
- **NVIDIA GPUs**: CUDA acceleration
- **Model Quantization**: INT8 and FP16 optimization
- **Mobile Export**: ONNX and Core ML deployment formats

## 📊 Performance Benchmarks

| Method | FPS (MPS) | FPS (CPU) | Model Size | PSNR | SSIM |
|--------|-----------|-----------|------------|------|------|
| CLAHE | ~30 | ~30 | - | 18.5 | 0.75 |
| Lightweight U-Net | ~25 | ~5 | 15MB | 24.2 | 0.89 |
| Standard U-Net | ~15 | ~2 | 30MB | 26.1 | 0.92 |
| Vision Transformer | ~10 | ~1 | 300MB | 25.8 | 0.91 |

*Benchmarks on 256x256 images using Apple M2 Pro*

## 📚 Datasets

### Supported Datasets
- **[LOL Dataset](https://daooshee.github.io/BMVC2018website/)**: Low-light paired images
- **[SID Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)**: See-in-the-Dark raw sensor data
- **[Meta Aria AEA](https://www.projectaria.com/datasets/aea/)**: Egocentric AR video data (optional)
- **Custom Data**: Webcam or custom low-light captures

### Data Preparation
```bash
# Download datasets to data/raw/
# Then preprocess:
python src/data/preprocess_lol.py --input_dir data/raw/LOL --output_dir data/processed/LOL
```

## 🏋️ Training Models

### Train U-Net
```bash
python src/training/train_unet.py --config configs/unet_config.yaml
```

### Train Vision Transformer
```bash
python src/training/train_unet.py --config configs/vit_config.yaml
```

## � Model Optimization

### Export for Deployment
```python
from src.models import create_model
from src.optimization.model_optimization import ModelOptimizer

# Load trained model
model = create_model('unet', 'lightweight')
model.load_state_dict(torch.load('models/checkpoints/best_checkpoint.pth'))

# Optimize for mobile
optimizer = ModelOptimizer(model)
export_paths = optimizer.optimize_for_mobile('exports/')
```

### Supported Export Formats
- **PyTorch**: Quantized models (.pth)
- **ONNX**: Cross-platform deployment (.onnx)
- **Core ML**: iOS/macOS deployment (.mlmodel)
- **TorchScript**: JIT compilation (.pt)

## 📊 Evaluation

### Comprehensive Evaluation
```python
from src.evaluation.metrics import EvaluationSuite

evaluator = EvaluationSuite(device='mps')
results = evaluator.evaluate_model(model, test_loader)
evaluator.print_results(results)
```

### Metrics Included
- **Quality**: PSNR, SSIM, LPIPS
- **Performance**: FPS, latency, memory usage
- **Efficiency**: Model size, energy consumption

## 🎮 Interactive Dashboard

The Streamlit dashboard provides:

1. **Real-time Enhancement**: Upload and enhance images instantly
2. **Method Comparison**: Side-by-side comparisons
3. **Parameter Tuning**: Interactive adjustment of enhancement parameters
4. **Performance Monitoring**: Real-time metrics and benchmarking
5. **Visualization**: Histograms, quality metrics, and detailed analysis

Launch with: `streamlit run dashboard/app.py`

## 🎯 Skills Demonstrated

- ✅ **Classical Computer Vision + Machine Learning**
- ✅ **Hardware-aware optimization** (MPS, ONNX, Core ML, quantization)
- ✅ **Image/video quality evaluation** (PSNR, SSIM, LPIPS)
- ✅ **Interactive tooling development**
- ✅ **PyTorch and Python programming**
- ✅ **AR/VR application relevance**
- ✅ **Meta Aria AEA dataset alignment**

## 🌟 Use Cases

### AR/VR Applications
- **Headset Feed Enhancement**: Real-time processing of camera feeds
- **Mixed Reality**: Improving visibility in challenging lighting
- **Computational Photography**: Enhanced capture in AR cameras

### Mobile Applications
- **Night Mode**: Smartphone camera enhancement
- **Video Calling**: Low-light video improvement
- **Security Cameras**: Surveillance in poor lighting

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LOL Dataset creators for low-light image pairs
- SID Dataset for raw sensor data
- Meta for the Aria Everyday Activities dataset
- PyTorch and open-source communities

---

⭐ **If you find this project useful, please give it a star!** ⭐
