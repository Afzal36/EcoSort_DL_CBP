
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path("models/waste_classifier.keras")
LABELS_PATH = Path("models/class_names.json")
IMAGE_SIZE = (224, 224)

WASTE_INFO = {
    "cardboard": {
        "icon": "📦",
        "advice": "Keep cardboard clean and dry. Flatten boxes before recycling.",
    },
    "glass": {
        "icon": "🍾",
        "advice": "Separate glass from other recyclables. Follow local rules for broken glass.",
    },
    "metal": {
        "icon": "🥫",
        "advice": "Empty metal cans and containers before placing them in recycling.",
    },
    "paper": {
        "icon": "📄",
        "advice": "Keep paper dry and free of food or oil contamination.",
    },
    "plastic": {
        "icon": "🧴",
        "advice": "Empty containers and check local recycling rules for the plastic type.",
    },
    "trash": {
        "icon": "🗑️",
        "advice": "This category is general trash. Check whether the item can be sorted into a recyclable stream.",
    },
}


st.set_page_config(
    page_title="EcoSort | Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f3fbf5 0%, #ffffff 45%);
    }
    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: #e0f4e5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #176b3a;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #285c3c;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>♻️ EcoSort</h1>
      <p>Smart Waste Classification Using Deep Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload a waste image to classify it into cardboard, glass, metal, "
    "paper, plastic, or trash."
)

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "The trained model or class labels are missing. "
        "Run `python train_model.py` and wait for training to finish."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


model = load_model()
class_names = load_labels()

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)

        # Use the same size and preprocessing as training.
        resized = image.resize(IMAGE_SIZE)
        image_array = np.asarray(resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        with st.spinner("Analyzing the image..."):
            probabilities = model.predict(image_array, verbose=0)[0]

        if len(probabilities) != len(class_names):
            st.error("Model output does not match the saved class labels.")
            st.stop()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        # Basic confidence-based rejection
        CONFIDENCE_THRESHOLD = 0.60

        if confidence < CONFIDENCE_THRESHOLD:
            with right:
                st.warning("Unknown / Unsupported Object")
                st.write(
                    "The model is not confident enough to classify this image "
                    "as cardboard, glass, metal, paper, plastic, or trash."
                )
        else:
            with right:
                info = WASTE_INFO.get(
                    predicted_class,
                    {"icon": "♻️", "advice": "Check local waste-sorting guidance."},
                )
                st.markdown(f"## {info['icon']} {predicted_class.title()}")
                st.metric("Model confidence", f"{confidence * 100:.1f}%")
                st.progress(confidence)
                st.info(f"**Disposal guidance:** {info['advice']}")

        with right:
            st.subheader("Prediction")
            info = WASTE_INFO.get(
                predicted_class,
                {"icon": "♻️", "advice": "Check local waste-sorting guidance."},
            )

            st.markdown(f"## {info['icon']} {predicted_class.title()}")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)

            if confidence < 0.50:
                st.warning(
                    "The model is uncertain about this image. "
                    "Try a clearer image with one waste item in focus."
                )

            st.info(f"**Disposal guidance:** {info['advice']}")

        st.divider()
        st.subheader("Confidence by category")

        chart_data = pd.DataFrame({
            "Waste category": [
                name.title() for name in class_names
            ],
            "Probability": [
                float(probability) for probability in probabilities
            ],
        }).sort_values("Probability", ascending=True)

        st.bar_chart(
            chart_data.set_index("Waste category"),
            horizontal=True,
        )

        st.caption(
            "Confidence scores are model outputs, not guarantees. "
            "Check local recycling guidelines before disposal."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.info("Upload an image to get a prediction.")

st.divider()
st.subheader("How EcoSort works")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("### 1. Upload")
    st.write("Choose an image of a waste item.")

with step2:
    st.markdown("### 2. Analyze")
    st.write("MobileNetV2 extracts visual features.")

with step3:
    st.markdown("### 3. Sort")
    st.write("View the predicted category and disposal guidance.")
