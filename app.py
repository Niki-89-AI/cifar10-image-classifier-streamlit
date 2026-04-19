import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")

# CIFAR-10 labels
LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@st.cache_resource
def load_trained_model():
    """
    Load the trained enhanced model once and cache it.
    Update the filename if your saved model uses a different name.
    """
    model = tf.keras.models.load_model("enhanced_cifar10_model.keras")
    return model

def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    """
    Convert uploaded image to RGB, resize to CIFAR-10 input size,
    normalize to [0, 1], and add batch dimension.
    """
    image = uploaded_image.convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image_array: np.ndarray):
    """
    Run inference and return predicted label, confidence,
    and probability vector.
    """
    probs = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probs))
    predicted_label = LABELS[predicted_index]
    confidence = float(probs[predicted_index])
    return predicted_label, confidence, probs

# UI
st.title("CIFAR-10 Image Classifier")
st.write(
    "Upload an image and the enhanced CNN model will predict one of the "
    "10 CIFAR-10 classes."
)

st.markdown(
    """
**Supported classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded image")
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        st.subheader("Prediction")
        try:
            model = load_trained_model()
            input_image = preprocess_image(image)
            predicted_label, confidence, probs = predict_image(model, input_image)

            st.success(f"Predicted class: **{predicted_label}**")
            st.write(f"Confidence: **{confidence:.2%}**")

            st.subheader("Class probabilities")
            prob_dict = {label: float(prob) for label, prob in zip(LABELS, probs)}
            st.bar_chart(prob_dict)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("FSE 560 Final Project - CIFAR-10 Image Classification with AI Interface")