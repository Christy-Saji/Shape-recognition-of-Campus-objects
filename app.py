import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# ==============================
# MUST BE FIRST STREAMLIT COMMAND
# ==============================
st.set_page_config(page_title="Shape Classification", layout="centered")

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 224
MODEL_PATH = "models/best_model.h5"
CONFIDENCE_THRESHOLD = 60  # %

# ==============================
# LOAD MODEL (Cached)
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# You can hardcode class names to ensure correct order
class_names = ["circle", "rectangle", "triangle"]

# ==============================
# TITLE
# ==============================
st.title("Shape Classification of Campus Objects")
st.write("Upload an image to classify it as Circle, Rectangle, or Triangle.")

# ==============================
# IMAGE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Load & preprocess image
    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict
    prediction = model.predict(img_array)
    probs = prediction[0] * 100
    confidence = np.max(probs)
    predicted_class = class_names[np.argmax(probs)]

    st.divider()

    # ==============================
    # CONFIDENCE CHECK
    # ==============================
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ The model is not confident. This may not be a valid geometric shape.")
        st.write(f"Highest Confidence: {confidence:.2f}%")
    else:
        st.success(f"Predicted Shape: **{predicted_class.upper()}**")
        st.info(f"Confidence: {confidence:.2f}%")

    # ==============================
    # PROBABILITY CHART
    # ==============================
    st.subheader("Prediction Probability Distribution")

    fig, ax = plt.subplots()
    ax.bar(class_names, probs)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Shape Class")
    st.pyplot(fig)

    # ==============================
    # INTERPRETATION SECTION
    # ==============================
    st.subheader("Model Interpretation")

    if predicted_class == "circle":
        st.write("The model detected round edges and symmetry typical of circular objects.")
    elif predicted_class == "rectangle":
        st.write("The model identified parallel edges and right-angle structure.")
    elif predicted_class == "triangle":
        st.write("The model recognized three edges forming a triangular structure.")