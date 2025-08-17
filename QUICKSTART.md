# Quick Start Guide

This guide will help you get started with the Real-Time Low-Light Image Enhancement project quickly.

## 🚀 Quick Setup (5 minutes)

### 1. Clone and Setup
```bash
cd Real-Time-Low-Light-Image-Enhancement-for-AR-VR
chmod +x setup.sh
./setup.sh
```

### 2. Activate Environment
```bash
source venv/bin/activate
```

### 3. Test Classical Enhancement (No training required)
```bash
# Download a sample low-light image or use your own
python src/classical/demo.py --input path/to/low_light_image.jpg --output_dir results/
```

### 4. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

## 🎯 Common Tasks

### Test Different Enhancement Methods
```bash
# Classical methods
python src/classical/demo.py --input data/samples/low_light.jpg --method clahe
python src/classical/demo.py --input data/samples/low_light.jpg --method gamma_correction
python src/classical/demo.py --input data/samples/low_light.jpg --method combined

# Compare all methods
python src/classical/demo.py --input data/samples/low_light.jpg --output_dir results/
```

### Train Your Own Model
```bash
# First, prepare your dataset (see data/README.md)
# Then train U-Net
python src/training/train_unet.py --config configs/unet_config.yaml

# Or train Vision Transformer
python src/training/train_unet.py --config configs/vit_config.yaml
```

### Optimize Models for Deployment
```python
from src.models import create_model
from src.optimization.model_optimization import ModelOptimizer

# Create and load model
model = create_model('unet', 'lightweight')
# Load trained weights: model.load_state_dict(torch.load('path/to/weights.pth'))

# Optimize for deployment
optimizer = ModelOptimizer(model)
export_paths = optimizer.optimize_for_mobile('exports/')
```

### Evaluate Model Performance
```python
from src.evaluation.metrics import EvaluationSuite

# Create evaluation suite
evaluator = EvaluationSuite(device='mps')  # or 'cuda', 'cpu'

# Evaluate model
results = evaluator.evaluate_model(model, test_loader)
evaluator.print_results(results)
```

## 📊 Using the Dashboard

The interactive dashboard provides:

1. **Real-time Enhancement**: Upload images and see results instantly
2. **Method Comparison**: Compare classical vs deep learning methods
3. **Performance Metrics**: See processing time, PSNR, SSIM scores
4. **Parameter Tuning**: Adjust enhancement parameters in real-time

Access at: http://localhost:8501 (after running `streamlit run dashboard/app.py`)

## 🎮 Interactive Examples

### Try with Sample Images
```bash
# Create some sample low-light images
python -c "
import cv2
import numpy as np

# Create a sample low-light image
img = np.random.randint(10, 50, (480, 640, 3), dtype=np.uint8)
cv2.imwrite('data/samples/sample_low_light.jpg', img)
print('Sample image created: data/samples/sample_low_light.jpg')
"

# Enhance it
python src/classical/demo.py --input data/samples/sample_low_light.jpg --output_dir results/
```

### Webcam Real-time Enhancement (Advanced)
```python
import cv2
from src.classical.enhancement import ClassicalEnhancer

# Initialize enhancer
enhancer = ClassicalEnhancer()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply enhancement
    enhanced, _ = enhancer.enhance_image(frame, 'combined')
    
    # Display
    cv2.imshow('Original', frame)
    cv2.imshow('Enhanced', enhanced)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🔧 Hardware Optimization

### For Apple Silicon (M1/M2)
```bash
# The project automatically uses MPS backend
# Verify MPS is working:
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### For NVIDIA GPUs
```bash
# Install CUDA version of PyTorch (uncomment in requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### For Mobile/Edge Deployment
```bash
# Export optimized models
python -c "
from src.models import create_model
from src.optimization.model_optimization import ModelOptimizer

model = create_model('unet', 'lightweight')
optimizer = ModelOptimizer(model)
paths = optimizer.optimize_for_mobile('exports/')
print('Exported models:', paths)
"
```

## 📈 Performance Expectations

| Method | FPS (CPU) | FPS (GPU) | Model Size | Quality |
|--------|-----------|-----------|------------|---------|
| CLAHE | ~30 | ~30 | - | Good |
| Gamma Correction | ~50 | ~50 | - | Fair |
| Lightweight U-Net | ~5 | ~25 | ~15MB | Excellent |
| Standard U-Net | ~2 | ~15 | ~30MB | Excellent |
| Vision Transformer | ~1 | ~10 | ~300MB | Excellent |

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated
   ```bash
   source venv/bin/activate
   ```

2. **CUDA Out of Memory**: Reduce batch size in configs
   ```yaml
   training:
     batch_size: 8  # Reduce from 16
   ```

3. **MPS Issues on Apple Silicon**: 
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

4. **Slow Performance**: Check device usage
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True for NVIDIA
   print(torch.backends.mps.is_available())  # Should be True for Apple Silicon
   ```

## 🎯 Next Steps

1. **Collect Data**: Download LOL/SID datasets for training
2. **Train Models**: Start with lightweight U-Net
3. **Optimize**: Export to ONNX/Core ML for deployment
4. **Evaluate**: Compare with baseline methods
5. **Deploy**: Integrate into AR/VR applications

## 📚 Resources

- [LOL Dataset](https://daooshee.github.io/BMVC2018website/)
- [SID Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)
- [Meta Aria AEA Dataset](https://www.projectaria.com/datasets/aea/)

Happy enhancing! 🌙✨
