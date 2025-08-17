#!/usr/bin/env python3
"""
AR/VR Low-Light Enhancement Demo
Streamlit interface for real-time enhancement
"""

import streamlit as st
import torch
import numpy as np
import cv2
import time
from PIL import Image
import plotly.graph_objects as go

# Import our enhancement methods
from classical_methods import ClassicalBaselines
from unet_model import CompactUNet
from vit_model import EnhancementViT

# Page config
st.set_page_config(
    page_title="AR/VR Low-Light Enhancement",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all enhancement models"""
    classical = ClassicalBaselines()
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    unet = CompactUNet(in_channels=3, out_channels=3).to(device)
    vit = EnhancementViT().to(device)
    
    return classical, unet, vit, device

def create_test_images():
    """Create sample low-light images"""
    images = {}
    
    # Dark indoor scene
    dark_indoor = np.random.randint(10, 60, (400, 600, 3), dtype=np.uint8)
    images['Dark Indoor'] = dark_indoor
    
    # Noisy night scene
    noisy_night = np.random.randint(5, 40, (400, 600, 3), dtype=np.uint8)
    noise = np.random.normal(0, 15, noisy_night.shape)
    noisy_night = np.clip(noisy_night + noise, 0, 255).astype(np.uint8)
    images['Noisy Night'] = noisy_night
    
    # AR/VR simulation
    arvr_sim = np.random.randint(15, 80, (400, 600, 3), dtype=np.uint8)
    images['AR/VR Feed'] = arvr_sim
    
    return images

def enhance_image(image, method, classical, unet, vit, device):
    """Apply enhancement method to image"""
    start_time = time.time()
    
    if method == "Original":
        result = image
    elif method == "CLAHE":
        result = classical.apply_clahe(image)
    elif method == "Bilateral":
        result = classical.apply_bilateral(image)
    elif method == "Gaussian":
        result = classical.apply_gaussian(image)
    elif method == "Combined":
        result = classical.apply_combined(image)
    elif method == "U-Net":
        result = enhance_with_unet(image, unet, device)
    elif method == "ViT":
        result = enhance_with_vit(image, vit, device)
    else:
        result = image
    
    processing_time = (time.time() - start_time) * 1000
    return result, processing_time

def enhance_with_unet(image, model, device):
    """Apply U-Net enhancement"""
    try:
        # Preprocess
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor = model(img_tensor)
        
        # Postprocess
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255).astype(np.uint8)
        return enhanced
    except Exception as e:
        st.error(f"U-Net error: {e}")
        return image

def enhance_with_vit(image, model, device):
    """Apply ViT enhancement"""
    try:
        # Resize for ViT
        img_resized = cv2.resize(image, (224, 224))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor = model(img_tensor)
        
        # Postprocess and resize back
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255).astype(np.uint8)
        enhanced = cv2.resize(enhanced, (image.shape[1], image.shape[0]))
        return enhanced
    except Exception as e:
        st.error(f"ViT error: {e}")
        return image

def calculate_metrics(original, enhanced):
    """Calculate basic quality metrics"""
    try:
        # PSNR
        mse = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Brightness improvement
        brightness_orig = np.mean(original)
        brightness_enh = np.mean(enhanced)
        brightness_improvement = (brightness_enh - brightness_orig) / brightness_orig * 100
        
        return {
            'PSNR': f"{psnr:.2f} dB",
            'Brightness Improvement': f"{brightness_improvement:.1f}%"
        }
    except:
        return {'PSNR': 'N/A', 'Brightness Improvement': 'N/A'}

def main():
    # Header
    st.title("AR/VR Low-Light Image Enhancement")
    st.markdown("Real-time enhancement for AR/VR headsets using classical and deep learning methods")
    
    # Load models
    classical, unet, vit, device = load_models()
    st.success(f"Models loaded successfully on {device}")
    
    # Sidebar controls
    st.sidebar.title("Enhancement Controls")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Enhancement Method:",
        ["Original", "CLAHE", "Bilateral", "Gaussian", "Combined", "U-Net", "ViT"]
    )
    
    # Input selection
    st.sidebar.subheader("Input Image")
    input_option = st.sidebar.radio(
        "Choose input:",
        ["Sample Images", "Upload Custom"]
    )
    
    # Get input image
    if input_option == "Sample Images":
        samples = create_test_images()
        sample_choice = st.sidebar.selectbox("Sample:", list(samples.keys()))
        input_image = samples[sample_choice]
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload image:", type=['jpg', 'jpeg', 'png']
        )
        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            input_image = np.array(pil_image)
            if len(input_image.shape) == 3 and input_image.shape[2] == 4:
                input_image = input_image[:, :, :3]  # Remove alpha
        else:
            input_image = create_test_images()['Dark Indoor']
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, caption="Low-light input", use_column_width=True)
        
        # Image statistics
        st.write(f"**Resolution:** {input_image.shape[1]} × {input_image.shape[0]}")
        st.write(f"**Mean Brightness:** {np.mean(input_image):.1f}")
    
    with col2:
        st.subheader(f"Enhanced ({method})")
        
        # Process image
        enhanced_image, proc_time = enhance_image(
            input_image, method, classical, unet, vit, device
        )
        
        st.image(enhanced_image, caption=f"Enhanced with {method}", 
                use_column_width=True)
        
        # Performance metrics
        fps = 1000 / proc_time if proc_time > 0 else 0
        realtime_status = "Real-time" if fps >= 30 else "Slow"
        
        st.write(f"**Processing Time:** {proc_time:.2f} ms")
        st.write(f"**FPS:** {fps:.1f} ({realtime_status})")
        
        # Quality metrics
        if method != "Original":
            metrics = calculate_metrics(input_image, enhanced_image)
            st.write("**Quality Metrics:**")
            for metric, value in metrics.items():
                st.write(f"  {metric}: {value}")
    
    # Comparison section
    st.markdown("---")
    st.subheader("Method Comparison")
    
    if st.button("Compare All Methods"):
        methods = ["CLAHE", "Bilateral", "Gaussian", "Combined", "U-Net", "ViT"]
        
        comparison_data = []
        progress_bar = st.progress(0)
        
        for i, test_method in enumerate(methods):
            enhanced, proc_time = enhance_image(
                input_image, test_method, classical, unet, vit, device
            )
            
            fps = 1000 / proc_time if proc_time > 0 else 0
            metrics = calculate_metrics(input_image, enhanced)
            
            comparison_data.append({
                'Method': test_method,
                'Time (ms)': f"{proc_time:.2f}",
                'FPS': f"{fps:.1f}",
                'Real-time': "Yes" if fps >= 30 else "No",
                'PSNR': metrics['PSNR']
            })
            
            progress_bar.progress((i + 1) / len(methods))
        
        # Results table
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Performance chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[row['Method'] for row in comparison_data],
            y=[float(row['Time (ms)']) for row in comparison_data],
            name='Processing Time (ms)'
        ))
        fig.update_layout(
            title="Processing Time Comparison",
            xaxis_title="Method",
            yaxis_title="Time (ms)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **AR/VR Low-Light Enhancement Demo**
    
    Features:
    - Classical Methods: CLAHE, Bilateral Filtering, Gaussian
    - Deep Learning: U-Net, Vision Transformer
    - Real-time Performance: Optimized for 30+ FPS
    - Quality Metrics: PSNR and brightness analysis
    """)

if __name__ == "__main__":
    main()
