import streamlit as st
import torch
import numpy as np
import cv2
import time
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pandas as pd

# Import our modules
try:
    from step1_classical_baselines import ClassicalEnhancer
    from model_inference import EnhancementEngine
    from evaluation_metrics import EvaluationMetrics
    classical_available = True
    ai_available = True
except ImportError as e:
    st.error(f"Import error: {e}")
    classical_available = False
    ai_available = False

# Page config
st.set_page_config(
    page_title="Real-Time Low-Light Enhancement for AR/VR",
    page_icon="🥽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
.method-section {
    border: 2px solid #E3F2FD;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background-color: #FAFAFA;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_enhancement_systems():
    """Load enhancement systems"""
    systems = {}
    
    if classical_available:
        try:
            systems['classical'] = ClassicalEnhancer()
        except:
            systems['classical'] = None
    
    if ai_available:
        try:
            systems['ai'] = EnhancementEngine()
            systems['metrics'] = EvaluationMetrics()
        except:
            systems['ai'] = None
            systems['metrics'] = None
    
    return systems

def create_sample_image(image_type="dark", size=(480, 640)):
    """Create sample test images"""
    h, w = size
    
    if image_type == "dark_scene":
        # Dark office/workspace scene
        img = np.random.randint(10, 50, (h, w, 3), dtype=np.uint8)
        # Add some structure
        cv2.rectangle(img, (50, 100), (w-50, h-50), (40, 35, 30), -1)
        cv2.rectangle(img, (100, 150), (200, 250), (20, 20, 25), -1)
        cv2.circle(img, (300, 200), 30, (60, 50, 40), -1)
        
    elif image_type == "noisy_lowlight":
        # Very dark with noise
        img = np.random.randint(5, 30, (h, w, 3), dtype=np.uint8)
        noise = np.random.normal(0, 15, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
    elif image_type == "ar_scene":
        # AR workplace simulation
        img = np.random.randint(20, 60, (h, w, 3), dtype=np.uint8)
        # Physical objects
        cv2.rectangle(img, (100, 200), (300, 400), (45, 40, 35), -1)
        cv2.circle(img, (450, 150), 40, (50, 45, 40), -1)
        # AR overlays (brighter)
        cv2.rectangle(img, (150, 100), (250, 150), (0, 255, 100), 3)
        cv2.putText(img, "AR Info", (160, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
        
    elif image_type == "vr_environment":
        # VR scene simulation
        img = np.random.randint(15, 45, (h, w, 3), dtype=np.uint8)
        # Virtual objects
        cv2.rectangle(img, (200, 150), (400, 350), (30, 25, 40), -1)
        cv2.circle(img, (150, 100), 25, (255, 100, 100), -1)
        cv2.circle(img, (500, 300), 35, (100, 100, 255), -1)
        
    else:  # bright reference
        img = np.random.randint(150, 255, (h, w, 3), dtype=np.uint8)
    
    return img

def process_with_method(image, method, enhancer_systems, params=None):
    """Process image with selected enhancement method"""
    if params is None:
        params = {}
    
    start_time = time.time()
    
    try:
        if method == "Original":
            result = image.copy()
        
        elif method == "CLAHE" and enhancer_systems.get('classical'):
            result = enhancer_systems['classical'].enhance_clahe(
                image, 
                clip_limit=params.get('clip_limit', 3.0),
                tile_grid_size=params.get('tile_grid_size', 8)
            )
        
        elif method == "Bilateral Filter" and enhancer_systems.get('classical'):
            result = enhancer_systems['classical'].enhance_bilateral_filter(
                image,
                d=params.get('d', 9),
                sigma_color=params.get('sigma_color', 75),
                sigma_space=params.get('sigma_space', 75)
            )
        
        elif method == "Histogram Equalization" and enhancer_systems.get('classical'):
            result = enhancer_systems['classical'].enhance_histogram_equalization(image)
        
        elif method == "Gaussian Filter" and enhancer_systems.get('classical'):
            result = enhancer_systems['classical'].enhance_gaussian_filter(
                image,
                kernel_size=params.get('kernel_size', 5),
                sigma=params.get('sigma', 1.0)
            )
        
        elif method == "U-Net" and enhancer_systems.get('ai'):
            result = enhancer_systems['ai'].enhance_with_unet(image)
        
        elif method == "ViT" and enhancer_systems.get('ai'):
            result = enhancer_systems['ai'].enhance_with_vit(image)
        
        else:
            result = image.copy()
            
    except Exception as e:
        st.error(f"Error processing with {method}: {e}")
        result = image.copy()
    
    processing_time = (time.time() - start_time) * 1000  # ms
    return result, processing_time

def calculate_image_metrics(original, enhanced, metrics_calc=None):
    """Calculate basic image quality metrics"""
    try:
        if metrics_calc and hasattr(metrics_calc, 'calculate_all_metrics'):
            metrics = metrics_calc.calculate_all_metrics(original, enhanced)
            return {
                'PSNR': f"{metrics.get('psnr', 0):.2f} dB",
                'SSIM': f"{metrics.get('ssim', 0):.4f}",
                'LPIPS': f"{metrics.get('lpips', 0):.4f}",
                'MAE': f"{metrics.get('mae', 0):.2f}"
            }
        else:
            # Basic metrics
            mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            brightness_original = np.mean(original)
            brightness_enhanced = np.mean(enhanced)
            brightness_improvement = ((brightness_enhanced - brightness_original) / brightness_original) * 100
            
            return {
                'PSNR': f"{psnr:.2f} dB",
                'Brightness Gain': f"{brightness_improvement:.1f}%",
                'Original Mean': f"{brightness_original:.1f}",
                'Enhanced Mean': f"{brightness_enhanced:.1f}"
            }
    except Exception as e:
        return {'Error': str(e)}

def main():
    # Main header
    st.markdown('<div class="main-header">🥽 Real-Time Low-Light Enhancement for AR/VR</div>', 
                unsafe_allow_html=True)
    
    # Load enhancement systems
    enhancer_systems = load_enhancement_systems()
    
    # Check system availability
    if not enhancer_systems.get('classical') and not enhancer_systems.get('ai'):
        st.error("Enhancement systems not available. Please check imports.")
        return
    
    # Sidebar controls
    st.sidebar.title("🎛️ Enhancement Controls")
    
    # Method selection
    available_methods = ["Original"]
    if enhancer_systems.get('classical'):
        available_methods.extend(["CLAHE", "Bilateral Filter", "Histogram Equalization", "Gaussian Filter"])
    if enhancer_systems.get('ai'):
        available_methods.extend(["U-Net", "ViT"])
    
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        available_methods
    )
    
    # Image type selection
    st.sidebar.subheader("📸 Input Image")
    image_type = st.sidebar.radio(
        "Choose image type:",
        ["Dark Scene", "Noisy Low-Light", "AR Scene", "VR Environment", "Upload Custom"]
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
            input_image = create_sample_image("dark_scene")
    else:
        type_mapping = {
            "Dark Scene": "dark_scene",
            "Noisy Low-Light": "noisy_lowlight", 
            "AR Scene": "ar_scene",
            "VR Environment": "vr_environment"
        }
        input_image = create_sample_image(type_mapping[image_type])
    
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
        st.subheader("📥 Original Image")
        st.image(input_image, caption="Input Image", use_column_width=True)
        
        # Original image stats
        st.markdown("**Image Statistics:**")
        st.write(f"• Dimensions: {input_image.shape[1]} × {input_image.shape[0]}")
        st.write(f"• Mean Brightness: {np.mean(input_image):.1f}")
        st.write(f"• Std Deviation: {np.std(input_image):.1f}")
        st.write(f"• Min/Max: {np.min(input_image)}/{np.max(input_image)}")
    
    with col2:
        st.subheader(f"✨ Enhanced Image ({enhancement_method})")
        
        # Process image
        enhanced_image, proc_time = process_with_method(
            input_image, enhancement_method, enhancer_systems, params
        )
        
        st.image(enhanced_image, caption=f"Enhanced with {enhancement_method}", 
                use_column_width=True)
        
        # Performance metrics
        fps = 1000 / proc_time if proc_time > 0 else float('inf')
        st.markdown("**Performance:**")
        st.write(f"• Processing Time: {proc_time:.2f} ms")
        st.write(f"• FPS: {fps:.1f}")
        
        # Real-time capability indicator
        if fps >= 30:
            st.success("✅ Real-time capable (30+ FPS)")
        elif fps >= 15:
            st.warning("⚠️ Near real-time (15-30 FPS)")
        else:
            st.error("❌ Not real-time (<15 FPS)")
    
    # Quality metrics section
    if enhancement_method != "Original":
        st.markdown("---")
        st.subheader("📊 Quality Metrics")
        
        metrics = calculate_image_metrics(
            input_image, enhanced_image, enhancer_systems.get('metrics')
        )
        
        # Display metrics in columns
        metric_cols = st.columns(len(metrics))
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(metric_name, metric_value)
    
    # Comparison section
    st.markdown("---")
    st.subheader("🔍 Method Comparison")
    
    if st.button("🚀 Compare All Methods", type="primary"):
        comparison_methods = [m for m in available_methods if m != "Original"]
        
        if not comparison_methods:
            st.warning("No enhancement methods available for comparison.")
            return
        
        comparison_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, method in enumerate(comparison_methods):
            status_text.text(f"Processing {method}...")
            
            enhanced, proc_time = process_with_method(
                input_image, method, enhancer_systems
            )
            
            metrics = calculate_image_metrics(
                input_image, enhanced, enhancer_systems.get('metrics')
            )
            
            fps = 1000 / proc_time if proc_time > 0 else float('inf')
            
            comparison_results.append({
                'Method': method,
                'Processing Time (ms)': f"{proc_time:.2f}",
                'FPS': f"{fps:.1f}",
                'Real-time': "Yes" if fps >= 30 else "No",
                **{k: v for k, v in metrics.items() if k != 'Error'}
            })
            
            progress_bar.progress((i + 1) / len(comparison_methods))
        
        status_text.text("Comparison complete!")
        
        # Display comparison table
        df = pd.DataFrame(comparison_results)
        st.dataframe(df, use_container_width=True)
        
        # Performance chart
        if len(comparison_results) > 0:
            fig = go.Figure()
            
            methods = [r['Method'] for r in comparison_results]
            times = [float(r['Processing Time (ms)']) for r in comparison_results]
            fps_values = [float(r['FPS']) for r in comparison_results]
            
            fig.add_trace(go.Bar(
                x=methods,
                y=times,
                name='Processing Time (ms)',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=methods,
                y=fps_values,
                mode='markers+lines',
                name='FPS',
                yaxis='y2',
                marker=dict(size=10, color='red')
            ))
            
            fig.update_layout(
                title="Performance Comparison",
                xaxis_title="Method",
                yaxis=dict(title="Processing Time (ms)", side="left"),
                yaxis2=dict(title="FPS", side="right", overlaying="y"),
                height=500,
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Real-time demo section
    st.markdown("---")
    st.subheader("🎥 Real-Time Demo Simulation")
    
    demo_cols = st.columns([1, 2, 1])
    with demo_cols[1]:
        if st.button("▶️ Start Real-Time Demo", type="secondary"):
            st.info("🎬 Simulating real-time processing with multiple frames...")
            
            # Create containers for the demo
            demo_container = st.empty()
            fps_container = st.empty()
            frame_container = st.empty()
            
            # Simulate processing multiple frames
            for frame_num in range(15):
                # Create a new test frame
                frame_types = ["dark_scene", "noisy_lowlight", "ar_scene"]
                frame_type = frame_types[frame_num % len(frame_types)]
                test_frame = create_sample_image(frame_type, size=(300, 400))
                
                # Process frame
                enhanced_frame, proc_time = process_with_method(
                    test_frame, enhancement_method, enhancer_systems, params
                )
                
                # Create side-by-side display
                combined_frame = np.hstack([test_frame, enhanced_frame])
                
                # Update display
                demo_container.image(
                    combined_frame,
                    caption=f"Frame {frame_num + 1}: Original (left) vs Enhanced (right)",
                    use_column_width=True
                )
                
                fps = 1000 / proc_time if proc_time > 0 else float('inf')
                fps_container.metric("Current FPS", f"{fps:.1f}")
                frame_container.text(f"Processing frame {frame_num + 1}/15...")
                
                time.sleep(0.2)  # Simulate frame delay
            
            st.success("✅ Real-time demo completed!")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ System Information")
    
    info_cols = st.columns(3)
    
    with info_cols[0]:
        st.markdown("**🎯 Available Methods:**")
        if enhancer_systems.get('classical'):
            st.write("✅ Classical Enhancement")
            st.write("  • CLAHE")
            st.write("  • Bilateral Filter") 
            st.write("  • Histogram Equalization")
            st.write("  • Gaussian Filter")
        else:
            st.write("❌ Classical Enhancement")
        
        if enhancer_systems.get('ai'):
            st.write("✅ AI Enhancement")
            st.write("  • U-Net")
            st.write("  • Vision Transformer")
        else:
            st.write("❌ AI Enhancement")
    
    with info_cols[1]:
        st.markdown("**⚡ Performance:**")
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        st.write(f"Device: {device.upper()}")
        st.write("Target: 30+ FPS")
        st.write("Real-time: < 33ms")
        
        st.markdown("**📐 Input Formats:**")
        st.write("• Images: JPG, PNG")
        st.write("• Resolution: Any")
        st.write("• Color: RGB")
    
    with info_cols[2]:
        st.markdown("**🎯 AR/VR Applications:**")
        st.write("• Mixed Reality")
        st.write("• Low-light Workspaces")
        st.write("• Outdoor AR")
        st.write("• VR Environments")
        st.write("• Real-time Video")
        
        st.markdown("**📊 Metrics:**")
        st.write("• PSNR (Quality)")
        st.write("• SSIM (Structure)")
        st.write("• Processing Speed")
        st.write("• Brightness Gain")

if __name__ == "__main__":
    main()
