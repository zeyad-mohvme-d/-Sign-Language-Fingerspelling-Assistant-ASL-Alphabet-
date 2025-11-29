import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Preprocessing imports
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess


# ============================
# 1. MODEL SETTINGS
# ============================

MODEL_PATHS = {
    "CNN": "Models\CNN_model.h5",
    "EfficientNetB0": "Models\EfficientNetB0_model.h5",
    "ResNet50": "Models\ResNet50_model.h5"
}

MODEL_INPUT_SIZES = {
    "CNN": (128, 128),
    "EfficientNetB0": (128, 128),
    "ResNet50": (224, 224),
}

PREPROCESS_FUNCTIONS = {
    "CNN": lambda x: x / 255.0,
    "EfficientNetB0": efficient_preprocess,
    "ResNet50": resnet_preprocess,
}


# ============================
# 2. LOAD MODEL FUNCTION
# ============================

@st.cache_resource
def load_selected_model(model_name):
    return tf.keras.models.load_model(MODEL_PATHS[model_name])


# ============================
# 3. STREAMLIT USER INTERFACE
# ============================

st.title("ASL Alphabet Classifier 🤟")
st.write("Choose a model and upload an image to classify.")

model_name = st.selectbox("Select Model:", ["CNN", "EfficientNetB0", "ResNet50"])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])


# ============================
# 4. PREDICTION LOGIC
# ============================

if uploaded_file is not None:
    # Load and show image
    st.image(uploaded_file, caption="Uploaded Image", width=250)

    # Load model
    model = load_selected_model(model_name)

    # Get correct size + preprocess function
    target_size = MODEL_INPUT_SIZES[model_name]
    preprocess_fn = PREPROCESS_FUNCTIONS[model_name]

    # Prepare image
    img = load_img(uploaded_file, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_fn(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction Result:")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

