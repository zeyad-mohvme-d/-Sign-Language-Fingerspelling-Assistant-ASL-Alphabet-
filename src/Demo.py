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
import time
import warnings
warnings.filterwarnings('ignore')

# ===========================
# Initialize Session State
# ===========================
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

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
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .camera-preview {
        border: 3px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background-color: #000;
        margin: 10px 0;
    }
    .captured-image-box {
        border: 3px solid #FF9800;
        border-radius: 10px;
        padding: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .webcam-button {
        background: linear-gradient(45deg, #4CAF50, #2E7D32) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        margin: 5px 0 !important;
    }
    .capture-button {
        background: linear-gradient(45deg, #FF5722, #D84315) !important;
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
# Webcam Functions with LIVE PREVIEW
# ===========================
def capture_with_preview():
    """Capture photo from webcam with live preview in Streamlit"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Please check camera permissions.")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    st.session_state.camera_active = True
    st.session_state.show_preview = True
    
    # Create placeholders
    preview_placeholder = st.empty()
    status_placeholder = st.empty()
    button_placeholder = st.empty()
    
    captured_image = None
    
    # Buttons
    col1, col2 = button_placeholder.columns(2)
    
    with col1:
        capture_btn = st.button("📸 CAPTURE NOW", key="live_capture", type="primary", use_container_width=True)
    
    with col2:
        cancel_btn = st.button("❌ CANCEL", key="cancel_capture", use_container_width=True)
    
    # Capture loop
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create display frame with annotations
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw green box for hand placement
        box_size = 300
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add instruction text
        cv2.putText(display_frame, "Place hand in green box", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'CAPTURE NOW' button", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert BGR to RGB for Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Show live preview
        preview_placeholder.image(display_frame_rgb, channels="RGB", 
                                 caption="Live Camera Preview - Place hand in green box",
                                 use_container_width=True)
        
        # Show status
        status_placeholder.info("🟢 Camera is active - Adjust your hand position")
        
        # Check for button presses
        if capture_btn:
            # Crop to green box area
            if y1 >= 0 and x1 >= 0 and y2 <= height and x2 <= width:
                captured_frame = frame[y1:y2, x1:x2]
            else:
                captured_frame = frame
            
            # Convert to RGB
            captured_image = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            
            # Save to session state
            st.session_state.captured_image = Image.fromarray(captured_image)
            st.session_state.camera_active = False
            
            # Show success message
            status_placeholder.success("✅ Photo captured successfully!")
            
            # Show captured image
            st.image(captured_image, caption="Captured Hand Sign", use_container_width=True)
            break
        
        elif cancel_btn:
            status_placeholder.warning("❌ Camera closed")
            st.session_state.camera_active = False
            break
        
        # Small delay to prevent high CPU usage
        time.sleep(0.03)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    st.session_state.camera_active = False
    st.session_state.show_preview = False
    
    return captured_image

# ===========================
# Normalization functions
# ===========================
def normalize_predictions(predictions):
    """Normalize predictions to get proper confidence scores"""
    predictions = np.array(predictions)
    
    if predictions.ndim == 2:
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        predictions = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    predictions = np.clip(predictions, 0, 1)
    return predictions

# ===========================
# Preprocessing Function
# ===========================
def prepare_image(image, model_name):
    """Prepare image for model prediction"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if model_name == "CNN (128x128)":
        target_size = (128, 128)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
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
        target_size = (128, 128)
        img = image.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array.reshape(1, 128, 128, 3), img_array

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

# Input method in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## 📷 Input Options")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["📁 Upload Image", "📸 Live Webcam"]
)

# Clear button
if st.sidebar.button("🗑️ Clear All", type="secondary"):
    st.session_state.captured_image = None
    st.session_state.camera_active = False
    st.session_state.prediction_made = False
    st.session_state.prediction_results = None
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
        
        elif input_method == "📸 Live Webcam":
            st.markdown("### Live Webcam Capture")
            
            if not st.session_state.camera_active and not st.session_state.captured_image:
                st.info("👆 Click 'Start Live Preview' to see camera feed")
                
                if st.button("🎥 Start Live Preview", type="primary", use_container_width=True):
                    # This will trigger the camera preview
                    st.session_state.camera_active = True
                    st.rerun()
            
            elif st.session_state.camera_active:
                # Show live preview section
                st.markdown('<div class="camera-preview">', unsafe_allow_html=True)
                st.markdown("#### 🔴 LIVE PREVIEW")
                capture_with_preview()
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show captured image if available
            if st.session_state.captured_image is not None:
                st.markdown('<div class="captured-image-box">', unsafe_allow_html=True)
                st.markdown("#### ✅ CAPTURED IMAGE")
                st.image(st.session_state.captured_image, 
                        caption="Your Hand Sign (Cropped from green box)", 
                        use_container_width=True)
                
                # Image actions
                col_act1, col_act2, col_act3 = st.columns(3)
                with col_act1:
                    if st.button("✅ Use This", type="primary", use_container_width=True):
                        image = st.session_state.captured_image
                        st.success("✅ Image selected for prediction")
                
                with col_act2:
                    if st.button("🔄 Retake", use_container_width=True):
                        st.session_state.captured_image = None
                        st.session_state.camera_active = True
                        st.rerun()
                
                with col_act3:
                    if st.button("🗑️ Discard", use_container_width=True):
                        st.session_state.captured_image = None
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                image = st.session_state.captured_image
        
        # Display ASL reference if no image
        if image is None:
            with st.expander("📚 Quick ASL Reference"):
                st.markdown("""
                **Common Letters:**
                - **A**: ✊ Fist with thumb alongside
                - **B**: ✋ Flat palm, fingers together  
                - **C**: C Curved fingers
                - **D**: 👆 Index finger up
                - **L**: 👌 OK sign
                - **Y**: 🤘 Rock on sign
                """)
    
    with col2:
        st.markdown("### 🤖 Model Selection & Prediction")
        
        if not models:
            st.error("❌ No models loaded. Please check model files.")
        else:
            model_choice = st.selectbox(
                "Select a model:",
                list(models.keys()),
                key="model_select"
            )
            
            # Determine which image to use
            current_image = None
            if input_method == "📁 Upload Image" and image is not None:
                current_image = image
            elif input_method == "📸 Live Webcam" and st.session_state.captured_image is not None:
                current_image = st.session_state.captured_image
            
            if current_image is not None:
                st.markdown("---")
                st.markdown("### 🚀 Ready to Predict!")
                
                # Show image preview
                st.image(current_image, width=250, caption="Image for Prediction")
                
                # Big prediction button
                if st.button("🎯 PREDICT NOW", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyzing hand sign..."):
                        try:
                            # Prepare image
                            img_ready, img_display = prepare_image(current_image, model_choice)
                            
                            # Get model and predict
                            model = models[model_choice]
                            prediction = model.predict(img_ready, verbose=0)
                            prediction_norm = normalize_predictions(prediction)
                            
                            # Get top 3 predictions
                            top_3_idx = np.argsort(prediction_norm[0])[-3:][::-1]
                            top_3_classes = [class_names[i] for i in top_3_idx]
                            top_3_confidences = [float(prediction_norm[0][i]) * 100 for i in top_3_idx]
                            
                            # Store results
                            st.session_state.prediction_results = {
                                'top_3_classes': top_3_classes,
                                'top_3_confidences': top_3_confidences,
                                'model_name': model_choice,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.prediction_made = True
                            
                            # Display results
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown("## 📊 Prediction Results")
                            st.markdown(f"### 🎯 **{top_3_classes[0]}**")
                            
                            confidence = top_3_confidences[0]
                            if confidence > 85:
                                emoji = "🟢"
                                color = "#4CAF50"
                            elif confidence > 70:
                                emoji = "🟡"
                                color = "#FF9800"
                            else:
                                emoji = "🔴"
                                color = "#F44336"
                            
                            st.markdown(f"**Confidence:** {emoji} <span style='color:{color}; font-size:24px; font-weight:bold;'>{confidence:.2f}%</span>", 
                                       unsafe_allow_html=True)
                            
                            st.progress(min(int(confidence), 100), text=f"{confidence:.2f}% confidence")
                            
                            st.markdown("#### Other Predictions:")
                            for i in range(1, 3):
                                col_a, col_b = st.columns([1, 3])
                                with col_a:
                                    st.markdown(f"**{i+1}. {top_3_classes[i]}**")
                                with col_b:
                                    conf = min(max(top_3_confidences[i], 0), 100)
                                    st.progress(int(conf), text=f"{top_3_confidences[i]:.2f}%")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Visualization
                            st.markdown("### 📈 Top 10 Predictions")
                            
                            top_10_idx = np.argsort(prediction_norm[0])[::-1][:10]
                            top_10_classes = [class_names[i] for i in top_10_idx]
                            top_10_confidences = [float(prediction_norm[0][i]) * 100 for i in top_10_idx]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(top_10_classes)))
                            bars = ax.barh(top_10_classes[::-1], top_10_confidences[::-1], color=colors)
                            
                            ax.set_xlabel('Confidence (%)')
                            ax.set_title(f'Top 10 Predictions - {model_choice}')
                            ax.set_xlim(0, 100)
                            
                            for bar, conf in zip(bars, top_10_confidences[::-1]):
                                width = bar.get_width()
                                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                       f'{conf:.1f}%', va='center', fontweight='bold')
                            
                            st.pyplot(fig)
                            
                            # Prediction details
                            with st.expander("🔍 View Detailed Analysis"):
                                st.write(f"**Model:** {model_choice}")
                                st.write(f"**Time:** {time.strftime('%H:%M:%S')}")
                                
                                # Show all predictions
                                st.markdown("**All Predictions (Top 15):**")
                                all_preds = []
                                for i in range(len(class_names)):
                                    all_preds.append({
                                        "Class": class_names[i],
                                        "Confidence": f"{prediction_norm[0][i] * 100:.2f}%"
                                    })
                                
                                all_preds.sort(key=lambda x: float(x['Confidence'].strip('%')), reverse=True)
                                
                                for i, pred in enumerate(all_preds[:15]):
                                    if i < 3:
                                        medal = ["🥇", "🥈", "🥉"][i]
                                        st.write(f"{medal} **{pred['Class']}**: {pred['Confidence']}")
                                    else:
                                        st.write(f"{i+1}. {pred['Class']}: {pred['Confidence']}")
                        
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")
            
            elif st.session_state.prediction_made and st.session_state.prediction_results:
                # Show previous prediction
                results = st.session_state.prediction_results
                
                st.markdown("## 📋 Previous Prediction")
                st.markdown(f"### Result: **{results['top_3_classes'][0]}**")
                st.markdown(f"#### Confidence: **{results['top_3_confidences'][0]:.2f}%**")
                st.markdown(f"*Model: {results['model_name']} | Time: {results['timestamp']}*")
                
                if st.button("🔄 Make New Prediction", type="secondary"):
                    st.session_state.prediction_made = False
                    st.session_state.prediction_results = None
                    st.rerun()
            
            else:
                st.info("👆 Capture or upload an image first, then click 'PREDICT NOW'")

# ===========================
# Page 2: Model Comparison
# ===========================
elif page == "📊 Model Comparison":
    st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
    
    st.info("🔍 Compare predictions across all models")
    
    # Input section
    comparison_image = None
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload image for comparison",
            type=["jpg", "jpeg", "png"],
            key="compare_upload"
        )
        if uploaded_file is not None:
            comparison_image = Image.open(uploaded_file).convert("RGB")
    
    elif input_method == "📸 Live Webcam":
        if st.session_state.captured_image is not None:
            comparison_image = st.session_state.captured_image
            st.image(comparison_image, caption="Captured Image", use_container_width=True)
        else:
            st.info("No captured image. Go to 'Single Model' page to capture first.")
    
    if comparison_image is not None and st.button("🔄 Compare All Models", type="primary", use_container_width=True):
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (model_name, model) in enumerate(models.items()):
            status_text.text(f"Processing {model_name}...")
            progress_bar.progress((idx + 1) / len(models))
            
            try:
                img_ready, _ = prepare_image(comparison_image, model_name)
                prediction = model.predict(img_ready, verbose=0)
                prediction_norm = normalize_predictions(prediction)
                
                pred_idx = np.argmax(prediction_norm[0])
                pred_class = class_names[pred_idx]
                confidence = float(prediction_norm[0][pred_idx]) * 100
                
                results.append({
                    "Model": model_name,
                    "Prediction": pred_class,
                    "Confidence": f"{confidence:.2f}%",
                    "Raw Confidence": confidence
                })
                
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
        
        status_text.text("✅ Comparison complete!")
        progress_bar.empty()
        
        # Display results
        st.markdown("### 📋 Comparison Results")
        
        results.sort(key=lambda x: x["Raw Confidence"], reverse=True)
        
        for i, result in enumerate(results):
            with st.container():
                cols = st.columns([1, 3, 2, 2])
                
                with cols[0]:
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."
                    st.markdown(f"**{rank_emoji}**")
                
                with cols[1]:
                    st.markdown(f"**{result['Model']}**")
                
                with cols[2]:
                    st.markdown(f"**{result['Prediction']}**")
                
                with cols[3]:
                    conf_color = "#4CAF50" if result["Raw Confidence"] > 80 else "#FF9800" if result["Raw Confidence"] > 60 else "#F44336"
                    st.markdown(f"<span style='color:{conf_color}; font-weight:bold;'>{result['Confidence']}</span>", 
                               unsafe_allow_html=True)
                
                st.progress(min(int(result["Raw Confidence"]), 100))

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
            # Mock accuracy
            accuracy = 0.85
            st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
    
    if selected_model:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Classes", "29")
        
        with col2:
            input_size = "128×128" if "128" in selected_model else "224×224"
            st.metric("Input Size", input_size)
        
        with col3:
            st.metric("Model Type", selected_model.split()[0])

# ===========================
# Footer
# ===========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>✋ ASL Alphabet Recognition System</h4>
    <p><strong>Live Webcam Preview • Instant Prediction • Multi-Model Comparison</strong></p>
    <p>See camera feed directly in browser • Green box for hand placement</p>
</div>
""", unsafe_allow_html=True)