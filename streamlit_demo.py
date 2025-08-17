#!/usr/bin/env python3
"""
Interactive AR/VR Low-Light Enhancement Demo
Streamlit dashboard with side-by-side comparisons, metrics, and sliders
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from step1_classical_baselines import ClassicalEnhancer
from model_inference import EnhancementEngine
from evaluation_metrics import EvaluationMetrics


# Page config
st.set_page_config(
    page_title="AR/VR Low-Light Enhancement",
    page_icon="🥽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.enhancement-section {
    border: 2px solid #E3F2FD;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load enhancement models with caching"""
    engine = EnhancementEngine()
    classical = ClassicalEnhancer()
    metrics = EvaluationMetrics()
    return engine, classical, metrics

def create_sample_image(img_type="dark"):
    """Create sample images for testing"""
    if img_type == "dark":
        # Create a dark image
        img = np.random.randint(0, 80, (400, 600, 3), dtype=np.uint8)
    elif img_type == "noisy":
        # Create a noisy dark image
        img = np.random.randint(0, 60, (400, 600, 3), dtype=np.uint8)
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    else:
        # Create a bright reference image
        img = np.random.randint(150, 255, (400, 600, 3), dtype=np.uint8)
    
    return img

def process_image(image, method, engine, classical, params=None):
    """Process image with selected method"""
    if params is None:
        params = {}
    
    start_time = time.time()
    
    if method == "Original":
        result = image
    elif method == "CLAHE":
        result = classical.enhance_clahe(image, **params)
    elif method == "Bilateral Filter":
        result = classical.enhance_bilateral_filter(image, **params)
    elif method == "Histogram Equalization":
        result = classical.enhance_histogram_equalization(image)
    elif method == "Gaussian Filter":
        result = classical.enhance_gaussian_filter(image, **params)
    elif method == "U-Net":
        result = engine.enhance_with_unet(image)
    elif method == "ViT":
        result = engine.enhance_with_vit(image)
    else:
        result = image
    
    processing_time = (time.time() - start_time) * 1000  # ms
    return result, processing_time

def calculate_metrics_for_display(original, enhanced, metrics_calc):
    """Calculate and format metrics for display"""
    try:
        metrics = metrics_calc.calculate_all_metrics(original, enhanced)
        return {
            'PSNR': f"{metrics.get('psnr', 0):.2f} dB",
            'SSIM': f"{metrics.get('ssim', 0):.4f}",
            'LPIPS': f"{metrics.get('lpips', 0):.4f}",
            'MAE': f"{metrics.get('mae', 0):.2f}"
        }
    except:
        return {'PSNR': 'N/A', 'SSIM': 'N/A', 'LPIPS': 'N/A', 'MAE': 'N/A'}

def main():
    # Header
    st.markdown('<div class="main-header">AR/VR Low-Light Enhancement Demo</div>', 
                unsafe_allow_html=True)
    
    # Load models
    engine, classical, metrics_calc = load_models()
    
    # Sidebar controls
    st.sidebar.title("Enhancement Controls")
    
    # Method selection
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        ["Original", "CLAHE", "Bilateral Filter", "Histogram Equalization", 
         "Gaussian Filter", "U-Net", "ViT"]
    )
    
    # Sample image selection
    st.sidebar.subheader("Input Image")
    image_type = st.sidebar.radio(
        "Choose sample image:",
        ["Dark Scene", "Noisy Low-Light", "Upload Custom"]
    )
    
    # Load input image
    if image_type == "Upload Custom":
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png']
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            input_image = np.array(image)
            if len(input_image.shape) == 3 and input_image.shape[2] == 4:
                input_image = input_image[:, :, :3]  # Remove alpha channel
        else:
            input_image = create_sample_image("dark")
    else:
        sample_type = "dark" if image_type == "Dark Scene" else "noisy"
        input_image = create_sample_image(sample_type)
    
    # Method-specific parameters
    params = {}
    if enhancement_method == "CLAHE":
        st.sidebar.subheader("CLAHE Parameters")
        params['clip_limit'] = st.sidebar.slider("Clip Limit", 1.0, 8.0, 3.0, 0.1)
        params['tile_grid_size'] = st.sidebar.slider("Tile Grid Size", 2, 16, 8, 1)
    
    elif enhancement_method == "Bilateral Filter":
        st.sidebar.subheader("Bilateral Filter Parameters")
        params['d'] = st.sidebar.slider("Diameter", 5, 15, 9, 2)
        params['sigma_color'] = st.sidebar.slider("Sigma Color", 10, 150, 75, 5)
        params['sigma_space'] = st.sidebar.slider("Sigma Space", 10, 150, 75, 5)
    
    elif enhancement_method == "Gaussian Filter":
        st.sidebar.subheader("Gaussian Filter Parameters")
        params['kernel_size'] = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
        params['sigma'] = st.sidebar.slider("Sigma", 0.5, 5.0, 1.0, 0.1)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, caption="Input Image", use_column_width=True)
        
        # Original image stats
        st.write(f"**Dimensions:** {input_image.shape[1]} × {input_image.shape[0]}")
        st.write(f"**Mean Brightness:** {np.mean(input_image):.1f}")
        st.write(f"**Std Deviation:** {np.std(input_image):.1f}")
    
    with col2:
        st.subheader(f"Enhanced Image ({enhancement_method})")
        
        # Process image
        enhanced_image, proc_time = process_image(
            input_image, enhancement_method, engine, classical, params
        )
        
        st.image(enhanced_image, caption=f"Enhanced with {enhancement_method}", 
                use_column_width=True)
        
        # Performance metrics
        fps = 1000 / proc_time if proc_time > 0 else 0
        st.write(f"**Processing Time:** {proc_time:.2f} ms")
        st.write(f"**FPS:** {fps:.1f}")
        
        # Quality metrics (only if method is not "Original")
        if enhancement_method != "Original":
            quality_metrics = calculate_metrics_for_display(
                input_image, enhanced_image, metrics_calc
            )
            
            st.write("**Quality Metrics:**")
            col2a, col2b = st.columns(2)
            with col2a:
                st.write(f"PSNR: {quality_metrics['PSNR']}")
                st.write(f"SSIM: {quality_metrics['SSIM']}")
            with col2b:
                st.write(f"LPIPS: {quality_metrics['LPIPS']}")
                st.write(f"MAE: {quality_metrics['MAE']}")
    
    # Comparison section
    st.markdown("---")
    st.subheader("Method Comparison")
    
    if st.button("Compare All Methods"):
        methods = ["CLAHE", "Bilateral Filter", "Histogram Equalization", 
                  "Gaussian Filter", "U-Net", "ViT"]
        
        comparison_results = []
        
        progress_bar = st.progress(0)
        for i, method in enumerate(methods):
            enhanced, proc_time = process_image(
                input_image, method, engine, classical
            )
            
            quality_metrics = calculate_metrics_for_display(
                input_image, enhanced, metrics_calc
            )
            
            comparison_results.append({
                'Method': method,
                'Processing Time (ms)': f"{proc_time:.2f}",
                'FPS': f"{1000/proc_time:.1f}" if proc_time > 0 else "∞",
                'PSNR': quality_metrics['PSNR'],
                'SSIM': quality_metrics['SSIM']
            })
            
            progress_bar.progress((i + 1) / len(methods))
        
        # Display comparison table
        import pandas as pd
        df = pd.DataFrame(comparison_results)
        st.dataframe(df, use_container_width=True)
        
        # Performance chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r['Method'] for r in comparison_results],
            y=[float(r['Processing Time (ms)']) for r in comparison_results],
            name='Processing Time (ms)'
        ))
        fig.update_layout(
            title="Processing Time Comparison",
            xaxis_title="Method",
            yaxis_title="Time (ms)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time demo section
    st.markdown("---")
    st.subheader("Real-Time Demo")
    
    if st.button("Start Webcam Demo (Simulated)"):
        st.info("In a real deployment, this would capture from webcam and process frames in real-time.")
        
        # Simulate real-time processing
        demo_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        for i in range(30):  # Simulate 30 frames
            # Create a random frame
            frame = create_sample_image("dark")
            
            # Process with selected method
            enhanced_frame, proc_time = process_image(
                frame, enhancement_method, engine, classical, params
            )
            
            # Display
            demo_placeholder.image(
                np.hstack([frame, enhanced_frame]), 
                caption=f"Frame {i+1}: Original (left) vs Enhanced (right)",
                use_column_width=True
            )
            
            fps = 1000 / proc_time if proc_time > 0 else 0
            fps_placeholder.write(f"Real-time FPS: {fps:.1f}")
            
            time.sleep(0.1)  # Simulate frame delay

if __name__ == "__main__":
    main()
        
    elif scene_type == "VR Street":
        # Street environment
        cv2.rectangle(img, (0, 200), (256, 256), (0.08, 0.08, 0.06), -1)
        cv2.rectangle(img, (20, 80), (80, 200), (0.05, 0.05, 0.07), -1)
        cv2.rectangle(img, (180, 60), (240, 200), (0.05, 0.05, 0.07), -1)
        cv2.circle(img, (50, 40), 6, (0.6, 0.5, 0.3), -1)
        cv2.circle(img, (210, 30), 6, (0.6, 0.5, 0.3), -1)
        
    else:  # AR Meeting
        # Conference room
        cv2.rectangle(img, (50, 180), (200, 230), (0.12, 0.10, 0.08), -1)
        cv2.rectangle(img, (100, 50), (150, 90), (0.03, 0.03, 0.05), -1)
        cv2.circle(img, (80, 150), 18, (0.2, 0.15, 0.12), -1)
        cv2.circle(img, (170, 150), 18, (0.15, 0.2, 0.15), -1)
        cv2.rectangle(img, (20, 20), (80, 50), (0.0, 0.3, 0.6), 2)
    
    # Apply low-light conditions
    dark_img = np.power(img, 2.0) * brightness
    noise_pattern = np.random.normal(0, noise_level, img.shape).astype(np.float32)
    noisy_img = dark_img + noise_pattern
    
    return np.clip(noisy_img, 0, 1)


def enhance_image(image, method, unet, device):
    """Enhance image with selected method."""
    start_time = time.time()
    
    if method == "CLAHE":
        enhanced = ClassicalEnhancers.clahe_enhancement(image)
    elif method == "Bilateral Filter":
        enhanced = ClassicalEnhancers.bilateral_filter(image)
    elif method == "Histogram EQ":
        enhanced = ClassicalEnhancers.histogram_equalization(image)
    elif method == "U-Net (AI)":
        with torch.no_grad():
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            enhanced_tensor = unet(tensor)
            enhanced = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    else:
        enhanced = image
    
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else float('inf')
    
    return enhanced, inference_time * 1000, fps


def calculate_metrics(original, enhanced):
    """Calculate simple quality metrics."""
    # Brightness improvement
    brightness_improvement = np.mean(enhanced) - np.mean(original)
    
    # Contrast improvement (std deviation)
    contrast_improvement = np.std(enhanced) - np.std(original)
    
    # Simple noise reduction metric
    noise_reduction = np.std(original - cv2.GaussianBlur((original * 255).astype(np.uint8), (3, 3), 0).astype(np.float32) / 255.0) - \
                     np.std(enhanced - cv2.GaussianBlur((enhanced * 255).astype(np.uint8), (3, 3), 0).astype(np.float32) / 255.0)
    
    return brightness_improvement, contrast_improvement, noise_reduction


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🥽 AR/VR Low-Light Enhancement Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time image enhancement for AR/VR headsets with classical and AI methods")
    
    # Load models and data
    unet, device, model_loaded = load_models()
    eval_results = load_evaluation_results()
    
    # Sidebar controls
    st.sidebar.markdown("## Enhancement Controls")
    
    # Scene selection
    scene_type = st.sidebar.selectbox(
        "Select AR/VR Scenario:",
        ["AR Workspace", "VR Street", "AR Meeting"]
    )
    
    # Lighting conditions
    st.sidebar.markdown("### Lighting Conditions")
    brightness = st.sidebar.slider("Brightness Level", 0.05, 0.3, 0.12, 0.01)
    noise_level = st.sidebar.slider("Noise Level", 0.02, 0.15, 0.08, 0.01)
    
    # Enhancement method
    st.sidebar.markdown("### Enhancement Method")
    method = st.sidebar.selectbox(
        "Choose Enhancement:",
        ["Original", "CLAHE", "Bilateral Filter", "Histogram EQ", "U-Net (AI)"]
    )
    
    # Generate test image
    test_image = generate_test_scene(scene_type, brightness, noise_level)
    
    # Main layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### Original (Low-Light)")
        st.image(test_image, caption=f"Scene: {scene_type}", use_column_width=True)
        
        # Original stats
        st.markdown(f"""
        <div class="metric-card">
        <strong>Original Stats:</strong><br>
        Brightness: {np.mean(test_image):.3f}<br>
        Contrast: {np.std(test_image):.3f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### ✨ Enhanced ({method})")
        
        if method != "Original":
            enhanced_image, inference_time, fps = enhance_image(test_image, method, unet, device)
            st.image(enhanced_image, caption=f"Method: {method}", use_column_width=True)
            
            # Enhanced stats
            brightness_imp, contrast_imp, noise_red = calculate_metrics(test_image, enhanced_image)
            
            realtime_status = "Real-time" if fps >= 30 else "Near real-time" if fps >= 24 else "Not real-time"
            
            st.markdown(f"""
            <div class="metric-card">
            <strong>Enhanced Stats:</strong><br>
            Brightness: {np.mean(enhanced_image):.3f} (+{brightness_imp*100:.1f}%)<br>
            Contrast: {np.std(enhanced_image):.3f} (+{contrast_imp*100:.1f}%)<br>
            Inference: {inference_time:.1f}ms ({fps:.1f} FPS)<br>
            Status: {realtime_status}
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.image(test_image, caption="Original Image", use_column_width=True)
    
    with col3:
        st.markdown("### Performance Metrics")
        
        if eval_results:
            # Create performance comparison chart
            methods = list(eval_results.keys())
            psnr_values = [eval_results[m]['psnr'] for m in methods]
            fps_values = [eval_results[m]['fps'] for m in methods]
            
            # PSNR chart
            fig_psnr = px.bar(
                x=methods, y=psnr_values,
                title="PSNR Comparison (Higher = Better)",
                labels={'x': 'Method', 'y': 'PSNR (dB)'}
            )
            fig_psnr.update_layout(height=300)
            st.plotly_chart(fig_psnr, use_container_width=True)
            
            # FPS chart
            fig_fps = px.bar(
                x=methods, y=fps_values,
                title="Performance Comparison (Higher = Better)",
                labels={'x': 'Method', 'y': 'FPS'}
            )
            fig_fps.update_layout(height=300)
            st.plotly_chart(fig_fps, use_container_width=True)
        else:
            st.info("Performance data will appear here after running the evaluation.")
    
    # Detailed comparison section
    st.markdown("---")
    st.markdown("## 🔍 Detailed Method Comparison")
    
    if st.button("Run All Methods Comparison"):
        comparison_results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        methods_to_test = ["CLAHE", "Bilateral Filter", "Histogram EQ", "U-Net (AI)"]
        
        for i, test_method in enumerate(methods_to_test):
            status_text.text(f"Testing {test_method}...")
            enhanced, inf_time, test_fps = enhance_image(test_image, test_method, unet, device)
            
            brightness_imp, contrast_imp, noise_red = calculate_metrics(test_image, enhanced)
            
            comparison_results[test_method] = {
                'image': enhanced,
                'inference_time': inf_time,
                'fps': test_fps,
                'brightness_improvement': brightness_imp,
                'contrast_improvement': contrast_imp,
                'noise_reduction': noise_red
            }
            
            progress_bar.progress((i + 1) / len(methods_to_test))
        
        status_text.text("Comparison complete!")
        
        # Display results
        cols = st.columns(len(methods_to_test))
        
        for i, (method, results) in enumerate(comparison_results.items()):
            with cols[i]:
                st.markdown(f"**{method}**")
                st.image(results['image'], use_column_width=True)
                
                realtime = "YES" if results['fps'] >= 30 else "NO"
                
                st.markdown(f"""
                **Performance:**
                - Time: {results['inference_time']:.1f}ms
                - FPS: {results['fps']:.1f} {realtime}
                - Brightness: +{results['brightness_improvement']*100:.1f}%
                - Contrast: +{results['contrast_improvement']*100:.1f}%
                """)
    
    # Information panels
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### About This Demo")
        st.markdown("""
        This interactive demo showcases real-time low-light image enhancement for AR/VR applications:
        
        **Classical Methods:**
        - CLAHE: Contrast Limited Adaptive Histogram Equalization
        - Bilateral Filter: Edge-preserving noise reduction
        - Histogram EQ: Global contrast enhancement
        
        **AI Method:**
        - U-Net: Deep learning enhancement with Noise2Noise training
        
        **Real-time Target:** 30+ FPS for AR/VR applications
        """)
    
    with col_info2:
        st.markdown("### 🛠️ Technical Details")
        st.markdown(f"""
        **System Configuration:**
        - Device: {device.type.upper()}
        - Model Status: {'Trained' if model_loaded else 'Untrained'}
        - U-Net Parameters: {sum(p.numel() for p in unet.parameters()):,}
        
        **Evaluation Metrics:**
        - PSNR: Peak Signal-to-Noise Ratio
        - SSIM: Structural Similarity Index
        - LPIPS: Learned Perceptual Image Patch Similarity
        - FPS: Frames Per Second
        
        **AR/VR Scenarios:**
        - Workspace: Mixed reality office environment
        - Street: VR passthrough navigation
        - Meeting: AR video conferencing
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### Project: Real-Time Low-Light Image Enhancement for AR/VR")
    st.markdown("**Skills Demonstrated:** Computer Vision • Deep Learning • PyTorch • Hardware Optimization • Interactive Demos")


if __name__ == "__main__":
    main()
