
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page config
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")

# CIFAR-10 labels
LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Folder containing one PNG per class (e.g. samples/airplane.png)
SAMPLES_DIR = Path(__file__).parent / "samples"

@st.cache_resource
def load_trained_model():
    """
    Load the trained enhanced model once and cache it.
    """
    model = tf.keras.models.load_model("enhanced_cifar10_model.keras")
    return model

@st.cache_data
def load_sample_images():
    """
    Load one sample image per class from the local `samples/` folder.
    Each file should be named `<label>.png` (e.g. samples/airplane.png).
    Returns a dict {label: uint8 numpy array of shape (32, 32, 3)}.
    """
    samples = {}
    for label in LABELS:
        path = SAMPLES_DIR / f"{label}.png"
        if path.exists():
            img = Image.open(path).convert("RGB").resize((32, 32))
            samples[label] = np.array(img, dtype=np.uint8)
    return samples

def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    """
    Convert image to RGB, resize to 32x32, normalize to [0, 1],
    and add a batch dimension.
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

# ---------- Sidebar: About ----------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
**FSE 560 Final Project**
*Nikoletta Biri*

A CNN image classifier trained on the CIFAR-10 dataset
(10 classes, 32×32 RGB images).
"""
    )

    st.subheader("Model architecture")
    st.markdown(
        """
Enhanced CNN (~1.2M parameters):

- **Block 1:** Conv(32) → BatchNorm → Conv(32) → MaxPool → Dropout(0.25)
- **Block 2:** Conv(64) → BatchNorm → Conv(64) → MaxPool → Dropout(0.25)
- **Block 3:** Conv(128) → BatchNorm → MaxPool → Dropout(0.25)
- **Head:** Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax)
"""
    )

    st.subheader("Training")
    st.markdown(
        """
- Optimizer: Adam (lr = 0.001)
- Loss: sparse categorical crossentropy
- Batch size: 32, up to 30 epochs
- EarlyStopping on val_loss (patience = 5)
"""
    )

    st.subheader("Enhancements over baseline")
    st.markdown(
        """
- **Dropout** regularization (0.25 after conv blocks, 0.5 before output)
- **Batch normalization** for faster, more stable training
- **Deeper architecture** (3 conv blocks, 32 → 64 → 128 filters)
- **Hyperparameter tuning** on batch size and dropout rates
- **EarlyStopping** on validation loss for better generalization
"""
    )

    st.subheader("Results")
    st.markdown(
        """
| Model              | Augmentation | Dropout    | Test Acc |
|--------------------|--------------|------------|----------|
| Baseline           | No           | No         | ~10%     |
| Enhanced (with aug)| Yes          | 0.25 / 0.5 | 0.79     |
| Enhanced (no aug)  | No           | 0.25 / 0.5 | **0.84** |

Deployed model: **Enhanced (no aug)**
Test accuracy: **84%** · Weighted F1: **0.84**
Best class: automobile (F1 0.93)
Hardest class: cat (F1 0.71)
"""
    )

# ---------- Main UI ----------
st.title("CIFAR-10 Image Classifier")
st.write(
    "Upload an image and the enhanced CNN model will predict one of the "
    "10 CIFAR-10 classes."
)
st.markdown(
    "**Supported classes:** airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship, truck"
)

# ---------- Sample images ----------
st.subheader("Try a sample image")
st.caption(
    "Click any class below to classify a real CIFAR-10 test image "
    "the model has never seen during training."
)

# Initialize session state for which image to classify
if "sample_image" not in st.session_state:
    st.session_state.sample_image = None
if "sample_label" not in st.session_state:
    st.session_state.sample_label = None

sample_images = load_sample_images()

if not sample_images:
    st.info(
        "Sample images unavailable: no files found in the `samples/` folder. "
        "You can still upload your own image below."
    )
else:
    cols = st.columns(5)
    for i, label in enumerate(LABELS):
        if label not in sample_images:
            continue
        with cols[i % 5]:
            st.image(
                sample_images[label],
                caption=label,
                use_container_width=True,
            )
            if st.button(f"Classify {label}", key=f"btn_{label}"):
                st.session_state.sample_image = sample_images[label]
                st.session_state.sample_label = label

st.markdown("---")

# ---------- File uploader ----------
st.subheader("Or upload your own image")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

# ---------- Decide which input to classify ----------
image_to_classify = None
input_caption = None

if uploaded_file is not None:
    image_to_classify = Image.open(uploaded_file)
    input_caption = "Uploaded image"
elif st.session_state.sample_image is not None:
    image_to_classify = Image.fromarray(st.session_state.sample_image)
    input_caption = f"Sample image (true label: {st.session_state.sample_label})"

# ---------- Prediction ----------
if image_to_classify is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input image")
        st.image(image_to_classify, caption=input_caption, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        try:
            model = load_trained_model()
            input_image = preprocess_image(image_to_classify)
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
