# Real-Time Low-Light Image Enhancement for AR/VR

A clean, focused implementation of low-light image enhancement using classical computer vision methods optimized for AR/VR applications.

## Features

- **Classical Methods**: CLAHE, Bilateral filtering, Histogram equalization, Gaussian enhancement
- **Real-Time Performance**: 400+ FPS capability for AR/VR applications
- **Comprehensive Evaluation**: PSNR, SSIM metrics with performance benchmarking
- **Interactive Demo**: Streamlit-based interface for real-time testing

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Classical Methods Evaluation**:
   ```bash
   python step1_classical_baselines.py
   ```

3. **Interactive Demo**:
   ```bash
   streamlit run streamlit_demo.py
   ```

## Project Structure

```
├── step1_classical_baselines.py    # Classical CV methods evaluation
├── streamlit_demo.py              # Interactive demonstration  
├── model_optimization.py          # Hardware optimization pipeline
├── models/                        # Trained model weights
└── requirements.txt               # Dependencies
```

## Performance Results

| Method | PSNR (dB) | SSIM | FPS | Real-time |
|--------|-----------|------|-----|-----------|
| CLAHE | 10.6 | 0.702 | 676 | ✓ |
| Bilateral | 11.4 | 0.339 | 414 | ✓ |
| Histogram EQ | 11.5 | 0.226 | 4211 | ✓ |
| Gaussian | 10.5 | 0.737 | 674 | ✓ |

## Hardware Requirements

- **Python 3.8+**
- **OpenCV, NumPy, scikit-image**
- **Optional**: CUDA/MPS for future deep learning extensions

## What This Demonstrates

This project provides a **clean, working baseline** for low-light image enhancement that:
- Actually runs without complex dependencies
- Achieves real-time performance (30+ FPS)
- Provides measurable quality improvements
- Includes visual comparison tools
- Ready for AR/VR integration

Perfect foundation for adding deep learning models later!

## License

MIT License - see LICENSE file for details.
