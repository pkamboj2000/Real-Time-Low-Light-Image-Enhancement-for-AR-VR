# Real-Time Low-Light Image Enhancement for AR/VR

Goal: This project aims to improve the clarity and quality of noisy or low-light images and videos from AR/VR headsets in real time, while keeping the solution energy efficient for on-device use.

The repository includes classical computer vision methods, deep learning models (U-Net and ViT), self-supervised training, and hardware optimization for AR/VR applications.

## Overview

Datasets:

Methods:
   - U-Net (fast, effective for enhancement)
   - Vision Transformer (ViT) variant (for benchmarking)
   - Self-supervised training with Noise2Noise

Hardware Efficiency:

Evaluation:

Demo / Tools:

Datasets:

Methods:
   - U-Net (fast, effective for enhancement)
   - Vision Transformer (ViT) variant (for benchmarking)
   - Self-supervised training with Noise2Noise

Hardware Efficiency:

Evaluation:

Demo / Tools:



- Comprehensive evaluation: PSNR, SSIM metrics with performance benchmarking
- Comprehensive evaluation: PSNR, SSIM metrics with performance benchmarking
- Interactive demo: Streamlit-based interface for real-time testing


## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train models (Apple Silicon GPU/MPS recommended):
   - All training scripts will use Apple Silicon GPU (MPS) if available, otherwise CPU.
   - To train U-Net (LOL dataset example):
     ```bash
     # Edit dataset_loaders.py to set your LOL dataset path
     python unet_model.py  # (add training entry point if not present)
     ```
   - To train ViT (SID dataset example):
     ```bash
     # Edit dataset_loaders.py to set your SID dataset path
## Project Structure

```
├── classical_methods.py          # Classical CV methods evaluation
├── demo.py                       # Interactive demonstration  
├── hardware_optimization.py      # Hardware optimization pipeline
├── evaluation.py                 # Comprehensive evaluation system
├── unet_model.py                 # U-Net architecture
├── vit_model.py                  # Vision Transformer implementation
├── noise2noise_training.py       # Self-supervised training
├── webcam_capture.py             # Custom dataset creation
├── dataset_loaders.py            # LOL/SID dataset loaders
└── requirements.txt              # Dependencies
```

## Performance Results

| Method | PSNR (dB) | SSIM | FPS | Real-time |
|--------|-----------|------|-----|-----------|
| CLAHE | 10.6 | 0.702 | 676 | Yes |
| Bilateral | 11.4 | 0.339 | 414 | Yes |
| Histogram EQ | 11.5 | 0.226 | 4211 | Yes |
| Gaussian | 10.5 | 0.737 | 674 | Yes |
- Python 3.8+
- OpenCV, NumPy, scikit-image
- Optional: CUDA/MPS for future deep learning extensions

- Provides measurable quality improvements
- Includes visual comparison tools
- Ready for AR/VR integration

Perfect foundation for adding deep learning models later!

## License

MIT License - see LICENSE file for details.
