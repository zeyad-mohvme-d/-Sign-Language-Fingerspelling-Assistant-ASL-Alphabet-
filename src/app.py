import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===========================
# Register custom preprocessing function
# ===========================
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def custom_preprocess_input(x):
    return efficient_preprocess(x)  # استخدم نفس الدالة اللي الموديل متحفظ بيها

# ===========================
# Load Models
# ===========================
models = {
    "CNN (128x128)": tf.keras.models.load_model("Models/cnn_model.h5"),
    "EfficientNetB0 (128x128)": tf.keras.models.load_model("Models/EfficientNetB0_model.h5"),
    "EfficientNetB0_V2 (128x128)": tf.keras.models.load_model(
        "Models/EfficientV2.keras",
        custom_objects={"preprocess_input": custom_preprocess_input}
    ),
    # "ResNet50 (224x224)": tf.keras.models.load_model("Models/ResNet50_model.h5")
}

# ===========================
# Class Names
# ===========================
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'SPACE','NOTHING','DELETE'
]

# ===========================
# Preprocessing Function
# ===========================
def prepare_image(image, model_name):
    if "CNN" in model_name:
        target_size = (128, 128)
        img = image.resize(target_size)
        img = np.array(img) / 255.0
        return img.reshape(1, 128, 128, 3)

    elif "EfficientNet" in model_name:
        target_size = (128, 128)
        img = image.resize(target_size)
        img = np.array(img)
        img = efficient_preprocess(img)
        return np.expand_dims(img, axis=0)

    elif "ResNet" in model_name:
        target_size = (224, 224)
        img = image.resize(target_size)
        img = np.array(img)
        img = resnet_preprocess(img)
        return np.expand_dims(img, axis=0)

# ===========================
# Streamlit UI
# ===========================
st.title("ASL Alphabet Recognition (CNN + ResNet50 + EfficientNetB0)")
st.write("Upload an image and choose a model to predict the ASL letter.")

model_choice = st.selectbox(
    "Select Model:",
    list(models.keys())
)

uploaded_file = st.file_uploader("Upload Hand Sign Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        st.write("🔍 Processing...")
        img_ready = prepare_image(image, model_choice)
        model = models[model_choice]

        prediction = model.predict(img_ready)
        predicted_index = np.argmax(prediction)
        predicted_letter = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        st.subheader(f"Predicted Letter: **{predicted_letter}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
