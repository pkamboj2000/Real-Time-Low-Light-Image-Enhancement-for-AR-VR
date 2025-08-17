"""
Streamlit dashboard for real-time low-light image enhancement.

Interactive web interface for testing and comparing different enhancement methods.
Provides real-time parameter tuning and side-by-side comparisons.

Author: Pranjal Kamboj
Created: August 2025
"""

import streamlit as st
import cv2
import numpy as np
import torch
import time
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project modules
try:
    from classical.enhancement import ClassicalEnhancer
    from models import create_model
    from evaluation.metrics import ImageQualityMetrics
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed correctly.")
    st.stop()


def load_image(uploaded_file):
    """Load and convert uploaded image for processing."""
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    return None


def apply_classical_enhancement(image, method, **kwargs):
    """Apply classical enhancement method with timing."""
    enhancer = ClassicalEnhancer()
    enhanced, processing_time = enhancer.enhance_image(image, method, **kwargs)
    return enhanced, processing_time


def apply_deep_learning_enhancement(image, model, device):
    """Apply deep learning enhancement."""
    # Preprocess image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Enhance
    start_time = time.time()
    with torch.no_grad():
        enhanced_tensor = model(image_tensor)
    processing_time = time.time() - start_time
    
    # Postprocess
    enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_np = (enhanced_np * 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
    
    return enhanced_bgr, processing_time


def calculate_metrics(original, enhanced):
    """Calculate image quality metrics."""
    metrics_calculator = ImageQualityMetrics()
    
    try:
        psnr = metrics_calculator.psnr(enhanced, original)
        ssim = metrics_calculator.ssim(enhanced, original)
        return {'PSNR': psnr, 'SSIM': ssim}
    except Exception as e:
        st.warning(f"Could not calculate metrics: {e}")
        return {'PSNR': 0, 'SSIM': 0}


def create_comparison_plot(original, enhanced, method_name):
    """Create side-by-side comparison plot."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original', f'Enhanced ({method_name})'),
        specs=[[{"type": "image"}, {"type": "image"}]]
    )
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    fig.add_trace(
        go.Image(z=original_rgb),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Image(z=enhanced_rgb),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Image Enhancement Comparison - {method_name}",
        showlegend=False,
        height=400
    )
    
    return fig


def create_histogram_comparison(original, enhanced):
    """Create histogram comparison."""
    # Convert to RGB for analysis
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Red Channel', 'Green Channel', 'Blue Channel')
    )
    
    colors = ['red', 'green', 'blue']
    
    for i, color in enumerate(colors):
        # Original histogram
        hist_orig, bins = np.histogram(original_rgb[:, :, i], bins=50, range=(0, 255))
        fig.add_trace(
            go.Scatter(
                x=bins[:-1], y=hist_orig,
                mode='lines', name=f'Original {color}',
                line=dict(color=color, dash='dash')
            ),
            row=1, col=i+1
        )
        
        # Enhanced histogram
        hist_enh, _ = np.histogram(enhanced_rgb[:, :, i], bins=50, range=(0, 255))
        fig.add_trace(
            go.Scatter(
                x=bins[:-1], y=hist_enh,
                mode='lines', name=f'Enhanced {color}',
                line=dict(color=color)
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Histogram Comparison",
        height=300,
        showlegend=True
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Low-Light Image Enhancement",
        page_icon="🌙",
        layout="wide"
    )
    
    st.title("🌙 Real-Time Low-Light Image Enhancement for AR/VR")
    st.markdown("Interactive dashboard for testing classical and deep learning enhancement methods")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # Method selection
    enhancement_type = st.sidebar.selectbox(
        "Enhancement Type",
        ["Classical Methods", "Deep Learning", "Comparison"]
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a low-light image to enhance"
    )
    
    # Load sample image if none uploaded
    if uploaded_file is None:
        st.sidebar.info("📁 Upload an image or use the sample below")
        # You can add sample images here
        sample_path = "data/samples/sample_low_light.jpg"
        if os.path.exists(sample_path):
            uploaded_file = sample_path
    
    # Main content
    if uploaded_file is not None:
        # Load image
        if isinstance(uploaded_file, str):
            image = cv2.imread(uploaded_file)
        else:
            image = load_image(uploaded_file)
        
        if image is None:
            st.error("Could not load image")
            return
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, caption="Original Low-Light Image")
        
        # Enhancement based on selected type
        if enhancement_type == "Classical Methods":
            st.sidebar.subheader("Classical Method Settings")
            
            method = st.sidebar.selectbox(
                "Method",
                ["combined", "clahe", "histogram_eq", "gamma_correction", 
                 "log_transform", "bilateral_filter", "unsharp_mask"]
            )
            
            # Method-specific parameters
            params = {}
            if method == "clahe":
                params['clip_limit'] = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0)
                params['tile_grid_size'] = (
                    st.sidebar.slider("Tile Grid Size", 4, 16, 8),
                    st.sidebar.slider("Tile Grid Size", 4, 16, 8)
                )
            elif method == "gamma_correction":
                params['gamma'] = st.sidebar.slider("Gamma", 0.1, 2.0, 0.5)
            elif method == "bilateral_filter":
                params['d'] = st.sidebar.slider("Diameter", 5, 15, 9)
                params['sigma_color'] = st.sidebar.slider("Sigma Color", 10, 150, 75)
                params['sigma_space'] = st.sidebar.slider("Sigma Space", 10, 150, 75)
            
            # Apply enhancement
            if st.sidebar.button("🚀 Enhance Image"):
                with st.spinner("Enhancing image..."):
                    enhanced, processing_time = apply_classical_enhancement(image, method, **params)
                
                with col2:
                    st.subheader("✨ Enhanced Image")
                    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    st.image(enhanced_rgb, caption=f"Enhanced using {method}")
                
                # Metrics
                st.subheader("📊 Metrics")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("Processing Time", f"{processing_time:.3f}s")
                
                # Calculate quality metrics (using enhanced as reference)
                try:
                    metrics = calculate_metrics(image, enhanced)
                    with metrics_col2:
                        st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
                    with metrics_col3:
                        st.metric("SSIM", f"{metrics['SSIM']:.4f}")
                except:
                    pass
                
                # Comparison plot
                st.subheader("🔍 Detailed Comparison")
                comparison_fig = create_comparison_plot(image, enhanced, method)
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Histogram comparison
                histogram_fig = create_histogram_comparison(image, enhanced)
                st.plotly_chart(histogram_fig, use_container_width=True)
        
        elif enhancement_type == "Deep Learning":
            st.sidebar.subheader("Deep Learning Settings")
            
            model_type = st.sidebar.selectbox("Model Type", ["UNet", "Vision Transformer"])
            model_variant = st.sidebar.selectbox("Variant", ["lightweight", "standard"])
            
            # Device selection
            device_options = ["cpu"]
            if torch.backends.mps.is_available():
                device_options.append("mps")
            if torch.cuda.is_available():
                device_options.append("cuda")
            
            device = st.sidebar.selectbox("Device", device_options)
            
            if st.sidebar.button("🤖 Load Model & Enhance"):
                try:
                    with st.spinner("Loading model..."):
                        # Create model
                        if model_type == "UNet":
                            model = create_model('unet', model_variant)
                        else:
                            model = create_model('vit', model_variant, img_size=min(image.shape[:2]))
                        
                        model = model.to(device)
                        model.eval()
                    
                    with st.spinner("Enhancing image..."):
                        enhanced, processing_time = apply_deep_learning_enhancement(image, model, device)
                    
                    with col2:
                        st.subheader("🤖 AI Enhanced Image")
                        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        st.image(enhanced_rgb, caption=f"Enhanced using {model_type} ({model_variant})")
                    
                    # Metrics
                    st.subheader("📊 Metrics")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Processing Time", f"{processing_time:.3f}s")
                        st.metric("FPS", f"{1/processing_time:.1f}")
                    
                    # Model info
                    param_count = sum(p.numel() for p in model.parameters())
                    with metrics_col2:
                        st.metric("Parameters", f"{param_count:,}")
                        st.metric("Model Size", f"{param_count * 4 / (1024**2):.1f} MB")
                    
                    # Comparison plot
                    st.subheader("🔍 AI Enhancement Comparison")
                    comparison_fig = create_comparison_plot(image, enhanced, f"{model_type} ({model_variant})")
                    st.plotly_chart(comparison_fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error during AI enhancement: {e}")
        
        elif enhancement_type == "Comparison":
            st.subheader("🏆 Method Comparison")
            
            if st.sidebar.button("🔄 Compare All Methods"):
                methods_to_compare = ["clahe", "gamma_correction", "bilateral_filter"]
                
                results = {}
                
                with st.spinner("Running comparison..."):
                    for method in methods_to_compare:
                        enhanced, proc_time = apply_classical_enhancement(image, method)
                        results[method] = {
                            'enhanced': enhanced,
                            'time': proc_time
                        }
                
                # Display results
                cols = st.columns(len(methods_to_compare))
                
                for i, (method, result) in enumerate(results.items()):
                    with cols[i]:
                        st.write(f"**{method.replace('_', ' ').title()}**")
                        enhanced_rgb = cv2.cvtColor(result['enhanced'], cv2.COLOR_BGR2RGB)
                        st.image(enhanced_rgb, use_column_width=True)
                        st.caption(f"Time: {result['time']:.3f}s")
                
                # Performance comparison chart
                st.subheader("⚡ Performance Comparison")
                
                methods = list(results.keys())
                times = [results[m]['time'] for m in methods]
                
                perf_fig = px.bar(
                    x=methods, y=times,
                    labels={'x': 'Method', 'y': 'Processing Time (s)'},
                    title="Processing Time Comparison"
                )
                st.plotly_chart(perf_fig, use_container_width=True)
    
    else:
        st.info("👆 Please upload an image to start enhancement")
        
        # Show instructions
        st.markdown("""
        ### 📋 Instructions
        
        1. **Upload Image**: Use the file uploader in the sidebar to upload a low-light image
        2. **Choose Method**: Select between Classical Methods, Deep Learning, or Comparison
        3. **Adjust Parameters**: Fine-tune the enhancement parameters in the sidebar
        4. **Enhance**: Click the enhance button to process your image
        5. **Analyze Results**: View the enhanced image, metrics, and detailed comparisons
        
        ### 🎯 Supported Methods
        
        **Classical Methods:**
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Histogram Equalization
        - Gamma Correction
        - Bilateral Filtering
        - And more...
        
        **Deep Learning Methods:**
        - U-Net (Standard and Lightweight variants)
        - Vision Transformer (Standard and Lightweight variants)
        
        ### 💡 Tips
        
        - Try different methods to see which works best for your image
        - Use the comparison mode to evaluate multiple methods at once
        - Monitor processing times for real-time applications
        - Check the metrics to quantify improvement
        """)


if __name__ == "__main__":
    main()
