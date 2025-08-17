#!/bin/bash

# Setup script for Real-Time Low-Light Image Enhancement for AR/VR
# Automated environment setup and dependency installation

set -e  # Exit on any error

echo "Setting up Real-Time Low-Light Image Enhancement for AR/VR"
echo "============================================================"
echo "This will install all required dependencies..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "  Python 3 is required but not installed."
    echo "   Please install from: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version (PyTorch requires 3.8+)
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "  Python 3.8+ is required. Found version $PYTHON_VERSION"
    echo "   Some dependencies require newer Python versions"
    exit 1
fi

echo "  Python $PYTHON_VERSION detected"

# Create virtual environment
echo "  Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "   Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,samples}
mkdir -p models/checkpoints
mkdir -p exports
mkdir -p results
mkdir -p logs

# Download sample data (optional)
echo "   Setting up sample data..."
# You can add commands here to download sample low-light images
# For now, we'll create a placeholder
touch data/samples/README.md
echo "Place sample low-light images in this directory for testing." > data/samples/README.md

# Set up Git hooks (optional)
echo "Setting up development tools..."
# You can add pre-commit hooks here if needed

# Check if GPU acceleration is available
echo "  Checking hardware acceleration..."
python3 -c "
import torch
print('  PyTorch installed successfully')
if torch.cuda.is_available():
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('  Apple Silicon MPS available')
else:
    print('   Running on CPU (consider GPU for better performance)')
"

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download datasets (see data/README.md for instructions)"
echo "3. Run the dashboard: streamlit run dashboard/app.py"
echo "4. Try classical enhancement: python src/classical/demo.py --help"
echo "5. Start training a model: python src/training/train_unet.py --config configs/unet_config.yaml"
echo ""
echo "For more information, see README.md"
