"""
Simple test script to verify project setup and basic functionality.
Run this after installation to ensure everything is working correctly.
"""

import os
import sys
import importlib


def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def test_imports():
    """Test critical package imports."""
    print("\n📦 Testing package imports...")
    
    critical_packages = [
        'numpy',
        'cv2',
        'PIL',
        'torch',
        'torchvision',
        'matplotlib',
        'streamlit',
        'yaml',
        'tqdm'
    ]
    
    optional_packages = [
        'lpips',
        'albumentations',
        'wandb',
        'onnx',
        'coremltools'
    ]
    
    all_good = True
    
    # Test critical packages
    for package in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING (Critical)")
            all_good = False
    
    # Test optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️ {package} - Missing (Optional)")
    
    return all_good


def test_torch_backends():
    """Test PyTorch hardware acceleration backends."""
    print("\n⚡ Testing PyTorch backends...")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        # Test CPU
        print("✅ CPU backend available")
        
        # Test CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA backend available: {device_name}")
        else:
            print("ℹ️ CUDA not available")
        
        # Test MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) backend available")
        else:
            print("ℹ️ MPS not available")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch backend test failed: {e}")
        return False


def test_project_structure():
    """Test project directory structure."""
    print("\n📁 Testing project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/classical',
        'src/data',
        'src/training',
        'src/evaluation',
        'src/optimization',
        'src/utils',
        'dashboard',
        'configs',
        'data',
        'notebooks'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - MISSING")
            all_good = False
    
    return all_good


def test_basic_functionality():
    """Test basic functionality without requiring datasets."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test classical enhancement
        sys.path.append('src')
        from classical.enhancement import ClassicalEnhancer
        
        import numpy as np
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test classical enhancer
        enhancer = ClassicalEnhancer()
        enhanced, proc_time = enhancer.enhance_image(dummy_image, 'gamma_correction')
        
        print(f"✅ Classical enhancement test passed ({proc_time:.3f}s)")
        
        # Test model creation (without training)
        try:
            from models import create_model
            model = create_model('unet', 'lightweight')
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ Model creation test passed ({param_count:,} parameters)")
        except Exception as e:
            print(f"⚠️ Model creation test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_config_files():
    """Test configuration files."""
    print("\n⚙️ Testing configuration files...")
    
    config_files = [
        'configs/unet_config.yaml',
        'configs/vit_config.yaml'
    ]
    
    all_good = True
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_file}")
            except Exception as e:
                print(f"❌ {config_file} - Invalid YAML: {e}")
                all_good = False
        else:
            print(f"❌ {config_file} - MISSING")
            all_good = False
    
    return all_good


def main():
    """Run all tests."""
    print("🌙 Real-Time Low-Light Image Enhancement - System Test")
    print("=" * 60)
    
    tests = [
        test_python_version,
        test_imports,
        test_torch_backends,
        test_project_structure,
        test_config_files,
        test_basic_functionality
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ Your setup is ready!")
        print("\nNext steps:")
        print("1. Download datasets (see data/README.md)")
        print("2. Launch dashboard: streamlit run dashboard/app.py")
        print("3. Try enhancement: python src/classical/demo.py --help")
    else:
        print(f"⚠️ {passed}/{total} tests passed")
        print("\n❌ Some issues found. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure virtual environment is activated")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check Python version (3.8+ required)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
