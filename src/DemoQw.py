import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.utils import register_keras_serializable
import cv2
import tempfile
import os
from datetime import datetime
import time
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# ===========================
# Initialize Session State
# ===========================
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'webcam_frame' not in st.session_state:
    st.session_state.webcam_frame = None
if 'camera_thread' not in st.session_state:
    st.session_state.camera_thread = None
if 'stop_camera' not in st.session_state:
    st.session_state.stop_camera = False

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="✋",
    layout="wide"
)

# ===========================
# Custom CSS
# ===========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #43A047;
        margin-top: 1.5rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .metric-box {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #C8E6C9;
        margin: 0.5rem 0;
    }
    .camera-container {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background-color: #000;
    }
    .webcam-button {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 10px 20px !important;
        border: none !important;
        border-radius: 5px !important;
        cursor: pointer !important;
        font-size: 16px !important;
        margin: 10px 0 !important;
        width: 100% !important;
    }
    .webcam-button:hover {
        background-color: #45a049 !important;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# Register custom preprocessing function
# ===========================
@register_keras_serializable()
def custom_preprocess_input(x):
    return efficient_preprocess(x)

# ===========================
# Load Models
# ===========================
@st.cache_resource
def load_models():
    try:
        models = {}
        
        # Load CNN Model
        try:
            models["CNN (128x128)"] = tf.keras.models.load_model("Models/CNN_model_V1.keras")
            st.sidebar.success("✓ CNN Model Loaded")
        except Exception as e:
            st.sidebar.error(f"✗ CNN Model: {str(e)[:50]}...")
        
        # Load EfficientNetB0 Model
        try:
            models["EfficientNetB0 (128x128)"] = tf.keras.models.load_model("Models/EfficientNetB0_model.h5")
            st.sidebar.success("✓ EfficientNetB0 Model Loaded")
        except Exception as e:
            st.sidebar.error(f"✗ EfficientNetB0: {str(e)[:50]}...")
        
        # Load EfficientNetB0_V2 Model
        try:
            models["EfficientNetB0_V2 (128x128)"] = tf.keras.models.load_model(
                "Models/EfficientV2.keras",
                custom_objects={"preprocess_input": custom_preprocess_input}
            )
            st.sidebar.success("✓ EfficientNetB0_V2 Model Loaded")
        except Exception as e:
            st.sidebar.error(f"✗ EfficientNetB0_V2: {str(e)[:50]}...")
        
        # Load ResNet50 Model
        try:
            models["ResNet50 (224x224)"] = tf.keras.models.load_model("Models/ResNet50_model_updated.keras")
            st.sidebar.success("✓ ResNet50 Model Loaded")
        except Exception as e:
            st.sidebar.error(f"✗ ResNet50: {str(e)[:50]}...")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

models = load_models()

# ===========================
# Class Names
# ===========================
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del', 'nothing', 'space'  
]

# ===========================
# Webcam Functions - SIMPLIFIED VERSION
# ===========================
def capture_single_photo():
    """Capture a single photo from webcam - SIMPLE VERSION"""
    cap = None
    try:
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check camera permissions.")
            return None
        
        # Set camera properties for faster capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Read a single frame
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            st.error("❌ Failed to capture frame")
            return None
            
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        return None
        
    finally:
        # Always release camera
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()

def simple_webcam_capture():
    """Simple one-click webcam capture"""
    if st.button("📸 Click to Capture Photo", key="simple_capture", use_container_width=True):
        with st.spinner("Capturing..."):
            captured_image = capture_single_photo()
            if captured_image is not None:
                st.session_state.captured_image = Image.fromarray(captured_image)
                st.success("✅ Photo captured successfully!")
                return Image.fromarray(captured_image)
    return None

# ===========================
# Normalization functions
# ===========================
def normalize_predictions(predictions):
    """
    Normalize predictions to get proper confidence scores between 0-100%
    """
    predictions = np.array(predictions)
    
    # If predictions are 2D (batch, classes)
    if predictions.ndim == 2:
        # Apply softmax if values are logits
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        predictions = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    # Ensure values are between 0 and 1
    predictions = np.clip(predictions, 0, 1)
    
    return predictions

# ===========================
# Preprocessing Function
# ===========================
def prepare_image(image, model_name):
    """
    Prepare image for model prediction
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
    
    if model_name == "CNN (128x128)":
        target_size = (128, 128)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32) 
        return img_array.reshape(1, 128, 128, 3), img_array
    
    elif model_name == "EfficientNetB0 (128x128)":
        target_size = (128, 128)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_array = efficient_preprocess(img_array)
        return img_array.reshape(1, 128, 128, 3), np.array(img)
    
    elif model_name == "EfficientNetB0_V2 (128x128)":
        target_size = (128, 128)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_array = efficient_preprocess(img_array)
        return img_array.reshape(1, 128, 128, 3), np.array(img)
    
    elif model_name == "ResNet50 (224x224)":
        target_size = (224, 224)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_array = resnet_preprocess(img_array)
        return img_array.reshape(1, 224, 224, 3), np.array(img)
    
    else:
        # Default preprocessing
        target_size = (128, 128)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array.reshape(1, 128, 128, 3), img_array

# ===========================
# Model Metrics (Mock Data)
# ===========================
def get_model_metrics(model_name):
    """Get accuracy and confusion matrix data for each model"""
    accuracies = {
        "CNN (128x128)": 0.85,
        "EfficientNetB0 (128x128)": 0.92,
        "EfficientNetB0_V2 (128x128)": 0.94,
        "ResNet50 (224x224)": 0.91
    }
    
    # Mock confusion matrix
    np.random.seed(42)
    num_classes = len(class_names)
    confusion_matrix = np.random.rand(num_classes, num_classes)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = confusion_matrix / row_sums
    
    return accuracies.get(model_name, 0.85), confusion_matrix

# ===========================
# Streamlit UI - Main Page
# ===========================
st.markdown('<h1 class="main-header">✋ ASL Alphabet Recognition System</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Single Model", "📊 Model Comparison", "📈 Model Performance"]
)

# Input method in sidebar - SIMPLIFIED
st.sidebar.markdown("---")
st.sidebar.markdown("## 📷 Input Options")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["📁 Upload Image", "📸 Webcam Photo"]
)

# Clear captured image button
if st.session_state.captured_image is not None:
    if st.sidebar.button("🗑️ Clear Captured Image"):
        st.session_state.captured_image = None
        st.rerun()

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Available Models")
for model_name in models.keys():
    st.sidebar.markdown(f"- {model_name}")

# ===========================
# Page 1: Single Model Prediction
# ===========================
if page == "🏠 Single Model":
    st.markdown('<h2 class="sub-header">Single Model Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Input Image")
        
        image = None
        
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an ASL hand sign image",
                type=["jpg", "jpeg", "png"],
                help="Upload an image of ASL alphabet sign",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    image = None
        
        elif input_method == "📸 Webcam Photo":
            st.markdown("### Webcam Capture")
            
            # Option 1: Use previously captured image
            if st.session_state.captured_image is not None:
                st.image(st.session_state.captured_image, caption="Captured Image", use_container_width=True)
                image = st.session_state.captured_image
                
                col_actions = st.columns(2)
                with col_actions[0]:
                    if st.button("✅ Use This Image", type="primary", use_container_width=True):
                        st.success("Using captured image for prediction")
                with col_actions[1]:
                    if st.button("🔄 Capture New", use_container_width=True):
                        st.session_state.captured_image = None
                        st.rerun()
            
            # Option 2: Capture new image
            else:
                st.info("Click the button below to capture a photo from your webcam")
                
                # Simple capture button
                captured_img = simple_webcam_capture()
                if captured_img:
                    image = captured_img
                    st.rerun()
                
                # Display instructions
                with st.expander("📸 Capture Instructions"):
                    st.markdown("""
                    **For best results:**
                    1. Make sure you have a webcam connected
                    2. Allow camera access when prompted
                    3. Position your hand clearly in frame
                    4. Ensure good lighting
                    5. Make a clear ASL sign
                    
                    **Common ASL signs:**
                    - **A:** Fist with thumb on side
                    - **B:** Flat hand, fingers together
                    - **C:** Curved hand like letter C
                    - **etc...**
                    """)
        
        # If no image, show sample
        if image is None and st.session_state.captured_image is None:
            st.info("👆 Please use one of the input methods above")
    
    with col2:
        st.markdown("### 🤖 Model Selection")
        
        if not models:
            st.error("❌ No models loaded. Please check the model files.")
        else:
            model_choice = st.selectbox(
                "Select a model for prediction:",
                list(models.keys()),
                help="Choose which trained model to use for prediction",
                key="model_select"
            )
            
            # Check if we have an image to predict
            current_image = None
            if input_method == "📁 Upload Image" and image is not None:
                current_image = image
            elif input_method == "📸 Webcam Photo" and st.session_state.captured_image is not None:
                current_image = st.session_state.captured_image
            
            if current_image is not None and st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                with st.spinner("🤖 Processing image..."):
                    try:
                        # Display the image being predicted
                        st.markdown("**Image for Prediction:**")
                        st.image(current_image, width=200)
                        
                        # Prepare image
                        img_ready, img_display = prepare_image(current_image, model_choice)
                        
                        # Get model and predict
                        model = models[model_choice]
                        prediction = model.predict(img_ready, verbose=0)
                        
                        # Normalize predictions
                        prediction_norm = normalize_predictions(prediction)
                        
                        # Get top 3 predictions
                        top_3_idx = np.argsort(prediction_norm[0])[-3:][::-1]
                        top_3_classes = [class_names[i] for i in top_3_idx]
                        top_3_confidences = [float(prediction_norm[0][i]) * 100 for i in top_3_idx]
                        
                        # Display results
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 Top 3 Predictions")
                        
                        # Top prediction with emphasis
                        st.markdown(f"### 🎯 **Top Prediction: {top_3_classes[0]}**")
                        
                        # Confidence gauge
                        confidence = top_3_confidences[0]
                        confidence_color = "#4CAF50" if confidence > 70 else "#FF9800" if confidence > 50 else "#F44336"
                        
                        st.markdown(f"**Confidence:** <span style='color:{confidence_color}; font-weight:bold;'>{confidence:.2f}%</span>", 
                                   unsafe_allow_html=True)
                        
                        # Progress bar for top prediction
                        st.progress(
                            min(int(confidence), 100), 
                            text=f"Confidence: {confidence:.2f}%"
                        )
                        
                        # Other predictions
                        st.markdown("#### Other Possible Predictions:")
                        for i in range(1, 3):
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.markdown(f"**{i+1}. {top_3_classes[i]}**")
                            with col_b:
                                conf = min(max(top_3_confidences[i], 0), 100)
                                st.progress(int(conf), text=f"{top_3_confidences[i]:.2f}%")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualize predictions as bar chart
                        st.markdown("### 📈 Prediction Distribution")
                        
                        # Get all predictions for chart
                        all_indices = np.argsort(prediction_norm[0])[::-1][:10]
                        all_classes = [class_names[i] for i in all_indices]
                        all_confidences = [float(prediction_norm[0][i]) * 100 for i in all_indices]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#4CAF50' if i == 0 else '#2196F3' for i in range(len(all_classes))]
                        bars = ax.barh(all_classes[::-1], all_confidences[::-1], color=colors[::-1])
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Top 10 Predictions')
                        ax.set_xlim(0, 100)
                        
                        # Add value labels
                        for bar, conf in zip(bars, all_confidences[::-1]):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{conf:.1f}%', va='center', fontweight='bold')
                        
                        st.pyplot(fig)
                        
                        # Prediction details
                        with st.expander("🔍 View Prediction Details"):
                            st.write("**Model Used:**", model_choice)
                            st.write("**Input Method:**", input_method)
                            st.write("**Timestamp:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            
                            # Raw prediction data
                            with st.expander("View Raw Data"):
                                st.write("**Raw prediction values (first 5):**", prediction[0][:5])
                                st.write("**Normalized values (first 5):**", prediction_norm[0][:5])
                                st.write("**Model input shape:**", img_ready.shape)
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.info("💡 **Troubleshooting tips:**")
                        st.markdown("""
                        1. Make sure the image clearly shows a hand sign
                        2. Try a different model
                        3. Check that model files are in 'Models' folder
                        4. Ensure image is in RGB format
                        """)

# ===========================
# Page 2: Model Comparison
# ===========================
elif page == "📊 Model Comparison":
    st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
    
    st.info("🔍 Use an image to compare predictions across all available models")
    
    # Input section for comparison
    comparison_image = None
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload image for comparison",
            type=["jpg", "jpeg", "png"],
            key="compare_upload"
        )
        if uploaded_file is not None:
            comparison_image = Image.open(uploaded_file).convert("RGB")
    
    elif input_method == "📸 Webcam Photo":
        st.markdown("#### Webcam Capture for Comparison")
        
        if st.session_state.captured_image is not None:
            comparison_image = st.session_state.captured_image
            st.image(comparison_image, caption="Captured Image for Comparison", use_container_width=True)
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("✅ Use This Image", key="use_for_compare", type="primary"):
                    st.success("Using captured image for comparison")
            with col_c2:
                if st.button("🔄 Capture New", key="new_for_compare"):
                    st.session_state.captured_image = None
                    st.rerun()
        else:
            st.info("Capture a photo first using the 'Single Model' page, or upload an image.")
    
    if comparison_image is not None:
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(comparison_image, caption="Test Image", use_container_width=True)
        
        if st.button("🔄 Compare All Models", type="primary", use_container_width=True):
            results = []
            predictions_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (model_name, model) in enumerate(models.items()):
                status_text.text(f"Processing {model_name}...")
                progress_bar.progress((idx + 1) / len(models))
                
                try:
                    # Prepare and predict
                    img_ready, _ = prepare_image(comparison_image, model_name)
                    prediction = model.predict(img_ready, verbose=0)
                    prediction_norm = normalize_predictions(prediction)
                    
                    # Get top prediction
                    pred_idx = np.argmax(prediction_norm[0])
                    pred_class = class_names[pred_idx]
                    confidence = float(prediction_norm[0][pred_idx]) * 100
                    
                    # Get top 3
                    top_3_idx = np.argsort(prediction_norm[0])[-3:][::-1]
                    top_3 = [(class_names[i], float(prediction_norm[0][i]) * 100) for i in top_3_idx]
                    
                    results.append({
                        "Model": model_name,
                        "Top Prediction": pred_class,
                        "Confidence": f"{confidence:.2f}%",
                        "Raw Confidence": confidence,
                        "Top 3": top_3
                    })
                    
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
                    results.append({
                        "Model": model_name,
                        "Top Prediction": "Error",
                        "Confidence": "0%",
                        "Raw Confidence": 0,
                        "Top 3": []
                    })
            
            status_text.text("✅ Comparison complete!")
            progress_bar.empty()
            
            # Display comparison results
            st.markdown("### 📋 Comparison Results")
            
            # Create a nice display
            for result in sorted(results, key=lambda x: x["Raw Confidence"], reverse=True):
                with st.container():
                    cols = st.columns([3, 2, 2, 3])
                    
                    with cols[0]:
                        st.markdown(f"**{result['Model']}**")
                    
                    with cols[1]:
                        pred_color = "#4CAF50" if result["Raw Confidence"] > 70 else "#FF9800"
                        st.markdown(f"<span style='color:{pred_color}; font-weight:bold;'>{result['Top Prediction']}</span>", 
                                   unsafe_allow_html=True)
                    
                    with cols[2]:
                        conf_color = "#4CAF50" if result["Raw Confidence"] > 70 else "#FF9800" if result["Raw Confidence"] > 50 else "#F44336"
                        st.markdown(f"<span style='color:{conf_color}; font-weight:bold;'>{result['Confidence']}</span>", 
                                   unsafe_allow_html=True)
                    
                    with cols[3]:
                        if result["Top 3"]:
                            badges = []
                            for i, (cls, conf) in enumerate(result["Top 3"][:3]):
                                badge_color = "#4CAF50" if i == 0 else "#2196F3" if i == 1 else "#9C27B0"
                                badges.append(f"<span style='background-color:{badge_color}; color:white; padding:2px 8px; border-radius:10px; margin:0 2px;'>{cls} ({conf:.0f}%)</span>")
                            st.markdown(" ".join(badges), unsafe_allow_html=True)
                    
                    st.progress(min(int(result["Raw Confidence"]), 100))
            
            # Find best model
            if results:
                best_result = max(results, key=lambda x: x["Raw Confidence"])
                st.success(f"🏆 **Best Model:** **{best_result['Model']}** predicted **{best_result['Top Prediction']}** with **{best_result['Confidence']}** confidence")

# ===========================
# Page 3: Model Performance
# ===========================
elif page == "📈 Model Performance":
    st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    col_select, col_info = st.columns([2, 1])
    
    with col_select:
        selected_model = st.selectbox(
            "Select model to view performance:",
            list(models.keys()),
            key="perf_select"
        )
    
    with col_info:
        if selected_model:
            accuracy, _ = get_model_metrics(selected_model)
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
            st.markdown(f'</div>', unsafe_allow_html=True)
    
    if selected_model:
        # Get metrics
        accuracy, confusion_matrix = get_model_metrics(selected_model)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Training Accuracy", f"{accuracy*100:.1f}%")
            st.markdown(f'</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Number of Classes", "29")
            st.markdown(f'</div>', unsafe_allow_html=True)
        
        with col3:
            input_size = "128×128" if "128" in selected_model else "224×224"
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Input Size", input_size)
            st.markdown(f'</div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("### 🤔 Confusion Matrix")
        
        # Let user choose how many classes to show
        num_classes_show = st.slider("Number of classes to display:", 5, 15, 10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(confusion_matrix[:num_classes_show, :num_classes_show],
                   annot=True, fmt='.2f',
                   cmap='Blues',
                   xticklabels=class_names[:num_classes_show],
                   yticklabels=class_names[:num_classes_show],
                   ax=ax)
        ax.set_title(f'Confusion Matrix - {selected_model} (First {num_classes_show} classes)')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        st.pyplot(fig)

# ===========================
# Footer
# ===========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>✋ ASL Alphabet Recognition System</h4>
    <p><strong>Features:</strong> Image Upload • Webcam Capture • Multi-Model Comparison • Performance Metrics</p>
    <p><strong>Models:</strong> CNN • EfficientNetB0 • ResNet50</p>
    <p><strong>Classes:</strong> 29 ASL signs (A-Z + del, nothing, space)</p>
</div>
""", unsafe_allow_html=True)

# Webcam troubleshooting
with st.sidebar.expander("🛠️ Webcam Help"):
    st.markdown("""
    **If webcam doesn't work:**
    1. Click "📸 Webcam Photo"
    2. Click "📸 Click to Capture Photo"
    3. Allow camera access when prompted
    4. Photo will be captured instantly
    
    **No live preview - instant capture only!**
    
    **For best results:**
    - Good lighting
    - Clear hand sign
    - Plain background
    - Center hand in frame
    """)