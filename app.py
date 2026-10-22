
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"
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

LAYER_STAGES = {
    "Input stem": {
        "layer": "Conv1",
        "short": "Stem",
        "shape": "112 x 112 x 32",
        "role": "The first convolution responds to local color changes and edges.",
        "kind": "spatial",
        "node": "stem",
    },
    "Texture": {
        "layer": "block_3_depthwise",
        "short": "Texture",
        "shape": "28 x 28 x 144",
        "role": "Depthwise filters respond to local texture and direction patterns.",
        "kind": "spatial",
        "node": "c3",
    },
    "Shape patterns": {
        "layer": "block_6_project",
        "short": "Shapes",
        "shape": "14 x 14 x 64",
        "role": "Compact feature maps combine lower-level responses into broader patterns.",
        "kind": "spatial",
        "node": "c4",
    },
    "Deep features": {
        "layer": "block_13_project",
        "short": "Deep features",
        "shape": "7 x 7 x 160",
        "role": "Later blocks encode coarser visual patterns; channels are learned responses, not named object parts.",
        "kind": "spatial",
        "node": "c5",
    },
    "Final feature map": {
        "layer": "Conv_1",
        "short": "Final map",
        "shape": "7 x 7 x 1280",
        "role": "The final MobileNetV2 map feeds global pooling and the waste classifier head.",
        "kind": "spatial",
        "node": "c5",
    },
    "C3 (28 x 28)": {
        "layer": "block_5_project",
        "short": "C3",
        "shape": "28 x 28 x 32",
        "role": "C3 is the higher-resolution feature stage, built from MobileNetV2 blocks 3 to 5.",
        "kind": "spatial",
        "node": "c3",
    },
    "C4 (14 x 14)": {
        "layer": "block_12_project",
        "short": "C4",
        "shape": "14 x 14 x 96",
        "role": "C4 is the mid-resolution feature stage, built from MobileNetV2 blocks 6 to 12.",
        "kind": "spatial",
        "node": "c4",
    },
    "C5 (7 x 7)": {
        "layer": "block_16_project",
        "short": "C5",
        "shape": "7 x 7 x 320",
        "role": "C5 is the deepest bottleneck stage, built from MobileNetV2 blocks 13 to 16.",
        "kind": "spatial",
        "node": "c5",
    },
    "Dense head": {
        "layer": "dense",
        "short": "Dense 128",
        "shape": "128 units",
        "role": "These learned combinations feed the six-way softmax classifier.",
        "kind": "dense",
        "node": "dense",
    },
}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_backbone():
    return next(
        layer for layer in load_model().layers
        if isinstance(layer, tf.keras.Model)
    )


@st.cache_resource
def get_activation_probe(layer_name):
    backbone = get_backbone()
    return tf.keras.Model(
        backbone.inputs,
        backbone.get_layer(layer_name).output,
    )


@st.cache_resource
def get_dense_probe():
    model = load_model()
    return tf.keras.Model(model.inputs, model.get_layer("dense").output)


@st.cache_resource
def get_gradcam_probe():
    backbone = get_backbone()
    return tf.keras.Model(
        backbone.inputs,
        [backbone.get_layer("Conv_1").output, backbone.output],
    )


def prepare_image(image):
    resized = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    return preprocess_input(np.expand_dims(image_array, axis=0))


@st.cache_data
def predict_image(image_tensor):
    return load_model().predict(image_tensor, verbose=0)[0]


@st.cache_data
def read_spatial_activations(layer_name, image_tensor):
    return get_activation_probe(layer_name)([image_tensor], training=False).numpy()[0]


@st.cache_data
def read_dense_activations(image_tensor):
    return get_dense_probe()([image_tensor], training=False).numpy()[0]


def normalize_map(values):
    values = np.nan_to_num(values.astype(np.float32))
    values = values - values.min()
    maximum = float(values.max())
    if maximum > 0:
        values = values / maximum
    return values


def activation_montage(activations, channel_limit=12):
    channel_count = min(activations.shape[-1], channel_limit)
    tile_size = 112
    label_height = 20
    padding = 8
    columns = 4
    rows = ceil(channel_count / columns)
    tile_height = tile_size + label_height
    canvas = Image.new(
        "RGB",
        (columns * (tile_size + padding), rows * (tile_height + padding)),
        (241, 246, 242),
    )
    draw = ImageDraw.Draw(canvas)

    for channel_index in range(channel_count):
        row, column = divmod(channel_index, columns)
        x = column * (tile_size + padding)
        y = row * (tile_height + padding)
        channel = normalize_map(activations[:, :, channel_index])
        tile = Image.fromarray((channel * 255).astype(np.uint8))
        tile = tile.resize(
            (tile_size, tile_size),
            resample=Image.Resampling.NEAREST,
        )
        canvas.paste(tile.convert("RGB"), (x, y + label_height))
        draw.text((x + 3, y + 3), f"CHANNEL {channel_index:02d}", fill=(36, 67, 53))

    return canvas


def activation_image(activation):
    normalized = normalize_map(activation)
    image = Image.fromarray((normalized * 255).astype(np.uint8))
    return image.resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)


def make_gradcam(image_tensor, class_index):
    model = load_model()
    with tf.GradientTape() as tape:
        convolution, backbone_output = get_gradcam_probe()(
            [image_tensor],
            training=False,
        )
        values = model.get_layer("global_average_pooling2d")(backbone_output)
        values = model.get_layer("batch_normalization")(
            values,
            training=False,
        )
        values = model.get_layer("dropout")(values, training=False)
        values = model.get_layer("dense")(values)
        values = model.get_layer("dropout_1")(values, training=False)
        scores = model.get_layer("dense_1")(values)
        score = scores[:, class_index]

    gradients = tape.gradient(score, convolution)
    if gradients is None:
        raise RuntimeError("Could not calculate gradients for this layer.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(convolution[0] * weights, axis=-1)
    return normalize_map(np.maximum(heatmap.numpy(), 0))


@st.cache_data
def read_gradcam(image_tensor, class_index):
    return make_gradcam(image_tensor, class_index)


def gradcam_overlay(image, heatmap):
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_image = heatmap_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    heat = np.asarray(heatmap_image, dtype=np.float32) / 255
    colors = np.stack(
        [255 * heat, 92 * (1 - heat), 24 * (1 - heat)],
        axis=-1,
    )
    base = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32)
    overlay = np.clip(base * 0.58 + colors * 0.42, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def architecture_markup(selected_stage):
    stage = LAYER_STAGES[selected_stage]
    nodes = [
        ("01 / INPUT", "224 x 224", "RGB image", "input"),
        ("02 / STEM", "Conv1", "112 x 112 x 32", "stem"),
        ("03 / C3", "28 x 28", "Blocks 3 to 5", "c3"),
        ("04 / C4", "14 x 14", "Blocks 6 to 12", "c4"),
        ("05 / C5", "7 x 7", "Blocks 13 to 16", "c5"),
        ("06 / HEAD", "Dense 128", "ReLU features", "dense"),
        ("07 / OUTPUT", "Softmax x 6", "Waste categories", "output"),
    ]
    elements = []
    for index, (eyebrow, title, detail, node_name) in enumerate(nodes):
        class_name = "flow-node active" if node_name == stage["node"] else "flow-node"
        elements.append(
            f'<div class="{class_name}"><span>{eyebrow}</span>'
            f'<strong>{title}</strong><small>{detail}</small></div>'
        )
        if index < len(nodes) - 1:
            elements.append('<div class="flow-arrow">&#8594;</div>')
    return '<div class="model-flow">' + "".join(elements) + "</div>"


st.set_page_config(
    page_title="EcoSort | Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #19372c;
        --muted: #68766e;
        --line: #d5ded7;
        --paper: #f5f7f3;
        --white: #ffffff;
        --leaf: #326a4c;
        --leaf-soft: #e2eee5;
        --citrus: #d7e66a;
        --coral: #dd774d;
        --blue: #4c8b9a;
    }
    .stApp {
        background-color: var(--paper);
        background-image: repeating-linear-gradient(0deg, transparent 0 31px, rgba(50, 106, 76, 0.035) 31px 32px);
        color: var(--ink);
        font-family: "Aptos", "Trebuchet MS", sans-serif;
    }
    .block-container { max-width: 1480px; padding: 1.5rem 2rem 3rem; }
    section[data-testid="stSidebar"] {
        background: #19372c;
        border-right: 1px solid #315443;
    }
    section[data-testid="stSidebar"] * { color: #edf2e9; }
    section[data-testid="stSidebar"] input[type="radio"] { accent-color: #d7e66a; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        min-height: 2.8rem;
        padding: 0.62rem 0.7rem;
        border: 1px solid transparent;
        border-radius: 6px;
        color: #dbe6dc;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #2b4b39;
        border-color: #52715b;
        border-left: 3px solid #d7e66a;
    }
    section[data-testid="stSidebar"] hr { border-color: #42604e; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 0.25rem; }
    .side-brand {
        padding: 0.45rem 0 1rem;
        border-bottom: 1px solid #42604e;
        margin-bottom: 0.85rem;
    }
    .side-brand strong {
        display: block;
        color: #f5f7ed;
        font-family: Georgia, serif;
        font-size: 1.45rem;
        font-weight: 500;
    }
    .side-brand span { color: #bfd0c2; font-size: 0.75rem; }
    .side-model {
        padding: 0.8rem;
        border: 1px solid #42604e;
        border-radius: 6px;
        background: #244434;
        margin: 0.8rem 0;
    }
    .side-model-kicker {
        color: #b7cbb9;
        font-size: 0.66rem;
        font-weight: 700;
    }
    .side-model-name {
        display: block;
        color: #f5f7ed;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .side-model-note { color: #cad8cb; font-size: 0.72rem; }
    .side-model-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.75rem;
        padding-top: 0.65rem;
        border-top: 1px solid #42604e;
    }
    .side-model-stats strong { display: block; color: #d7e66a; font-size: 1.15rem; }
    .side-model-stats span { color: #c2d1c4; font-size: 0.68rem; }
    .side-category-title { color: #b7cbb9; font-size: 0.66rem; font-weight: 700; }
    .side-category-list {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.45rem 0.35rem;
        margin-top: 0.55rem;
    }
    .side-category-list span { color: #e2ebe2; font-size: 0.72rem; }
    .side-category-list span::before { content: ""; display: inline-block; width: 6px; height: 6px; margin: 0 0.38rem 1px 0; border-radius: 50%; background: #d7e66a; }
    .masthead {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.35rem 0 1.1rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.2rem;
    }
    .masthead-title {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.8rem;
        line-height: 1.1;
    }
    .masthead-subtitle { color: var(--muted); font-size: 0.86rem; margin-top: 0.3rem; }
    .ready-mark {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.7rem;
        border: 1px solid #bdd2c1;
        border-radius: 5px;
        background: #e8f1e9;
        color: #285a3d;
        font-size: 0.75rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .ready-dot { width: 8px; height: 8px; border-radius: 50%; background: #4c9a62; }
    .page-kicker {
        color: var(--leaf);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .upload-ribbon {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.85rem 1rem;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: var(--white);
        margin: 0.5rem 0 1.2rem;
    }
    .upload-ribbon strong { color: var(--ink); font-size: 0.9rem; }
    .upload-ribbon span { color: var(--muted); font-size: 0.76rem; }
    .result-band {
        padding: 1.1rem 1.2rem;
        border-left: 4px solid var(--coral);
        border-radius: 0 7px 7px 0;
        background: #fff5ed;
        margin: 0.8rem 0 1rem;
    }
    .result-band strong { color: var(--ink); }
    .subtle-rule { border-top: 1px solid var(--line); margin: 1.4rem 0; }
    .hero {
        padding: 1.1rem 1.3rem;
        border: 1px solid #cfdbd0;
        border-left: 5px solid var(--leaf);
        border-radius: 7px;
        background: linear-gradient(105deg, #e8f0e7 0%, #f8faf6 76%);
        margin-bottom: 0.85rem;
    }
    .hero h1 {
        color: var(--ink);
        margin: 0;
        font-size: 1.85rem;
        font-family: Georgia, "Times New Roman", serif;
    }
    .hero p {
        color: var(--muted);
        margin: 0.3rem 0 0;
        font-size: 0.84rem;
    }
    .eyebrow {
        color: var(--leaf);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .model-flow {
        display: flex;
        align-items: stretch;
        gap: 0.35rem;
        margin: 0.55rem 0 0.9rem;
        overflow-x: auto;
        padding: 0.4rem 0 0.65rem;
    }
    .flow-node {
        display: flex;
        flex: 1 0 128px;
        min-height: 92px;
        flex-direction: column;
        justify-content: center;
        gap: 0.38rem;
        padding: 0.72rem;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--white);
    }
    .flow-node.active {
        border: 2px solid var(--leaf);
        background: var(--leaf-soft);
    }
    .flow-node span {
        color: var(--muted);
        font-size: 0.65rem;
        font-weight: 700;
    }
    .flow-node strong { color: var(--ink); font-size: 0.9rem; }
    .flow-node small { color: var(--muted); font-size: 0.7rem; }
    .flow-arrow {
        align-self: center;
        color: #83948a;
        font-size: 1.15rem;
    }
    [data-testid="stMetric"] {
        padding: 0.7rem 0.85rem;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--white);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: var(--white);
        border: 1px dashed #96aa9b;
        color: var(--ink);
        border-radius: 6px;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] button {
        background: var(--leaf);
        color: #ffffff;
        border: 0;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--leaf); }
    .stTabs button[aria-selected="true"] { color: var(--leaf); }
    .stAlert {
        background: #e8f2e9;
        border-color: #c5d8c9;
        color: var(--ink);
    }
    .stage-note {
        padding: 0.8rem 1rem;
        border-left: 3px solid var(--coral);
        background: #fff8f1;
        color: #5c4a3b;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0 1rem;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        .masthead { align-items: flex-start; }
        .masthead-title { font-size: 1.5rem; }
        .ready-mark { font-size: 0.68rem; }
        .hero { padding: 0.9rem; }
        .hero h1 { font-size: 1.55rem; }
        .flow-node { flex-basis: 116px; }
        .upload-ribbon { align-items: flex-start; flex-direction: column; }
        .side-model { padding: 0.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
        <div class="hero">
            <div class="eyebrow">WASTE CLASSIFICATION LAB</div>
            <h1>EcoSort</h1>
            <p>Classify an item, then inspect the network that made the prediction.</p>
        </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "The trained model or class labels are missing. "
        "Run `python train_model.py` and wait for training to finish."
    )
    st.stop()


model = load_model()
class_names = load_labels()

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
          <strong>EcoSort</strong>
          <span>WASTE CLASSIFICATION LAB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_view = st.radio(
        "Workspace",
        ["Classify", "Model explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"""
        <div class="side-model">
          <div class="side-model-kicker">MODEL CARD</div>
          <span class="side-model-name">MobileNetV2</span>
          <span class="side-model-note">Transfer-learned classifier</span>
          <div class="side-model-stats">
            <div><strong>{len(class_names):02d}</strong><span>classes</span></div>
            <div><strong>224</strong><span>input pixels</span></div>
          </div>
        </div>
        <div class="side-category-title">SORTING CATEGORIES</div>
        <div class="side-category-list">
          <span>Cardboard</span><span>Glass</span><span>Metal</span>
          <span>Paper</span><span>Plastic</span><span>Trash</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

st.markdown(
    """
    <div class="masthead">
      <div>
        <div class="page-kicker">SORTING INTELLIGENCE / FIELD TOOL</div>
        <div class="masthead-title">Waste, understood.</div>
        <div class="masthead-subtitle">Classify a material or inspect the features behind a prediction.</div>
      </div>
      <div class="ready-mark"><span class="ready-dot"></span> MODEL READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="upload-ribbon"><div><strong>IMAGE INPUT</strong><br>'
    '<span>One clear item works best · JPG, PNG, or WEBP</span></div>'
    '<span>224 x 224 model input</span></div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drop an image here or browse files",
    type=["jpg", "jpeg", "png", "webp"],
    help="The same image powers the prediction, feature maps, and Grad-CAM view.",
    label_visibility="collapsed",
)

image = None
image_tensor = None
probabilities = None
predicted_index = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = prepare_image(image)
        with st.spinner("Running the classifier..."):
            probabilities = predict_image(image_tensor)
        if len(probabilities) != len(class_names):
            st.error("Model output does not match the saved class labels.")
            st.stop()
        predicted_index = int(np.argmax(probabilities))
    except Exception as error:
        st.error(f"Could not process this image: {error}")
        st.stop()

if selected_view == "Classify":
    st.markdown('<div class="page-kicker">01 / MATERIAL SORTING</div>', unsafe_allow_html=True)
    st.header("Classification")
    if image is None:
        st.markdown(
            """
            <div class="hero">
              <div class="eyebrow">READY FOR AN IMAGE</div>
              <h1>Start with one item.</h1>
              <p>EcoSort will compare it against six material categories and show its confidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Available categories**")
        st.write("Cardboard · Glass · Metal · Paper · Plastic · Trash")
    else:
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        ranked_indices = np.argsort(probabilities)[::-1]
        second_class = class_names[int(ranked_indices[1])]
        confidence_gap = confidence - float(probabilities[ranked_indices[1]])

        st.markdown(
            f'<div class="result-band"><strong>Top match: {predicted_class.title()}</strong>'
            f' · {confidence:.1%} confidence · next match: {second_class.title()}</div>',
            unsafe_allow_html=True,
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric("Predicted material", predicted_class.title())
        metric_columns[1].metric("Confidence", f"{confidence:.1%}")
        metric_columns[2].metric("Lead over next class", f"{confidence_gap:.1%}")

        photo_column, analysis_column = st.columns([1, 1.15], gap="large")
        with photo_column:
            st.markdown('<div class="page-kicker">SOURCE IMAGE</div>', unsafe_allow_html=True)
            st.image(image, caption="Uploaded material", width="stretch")

        with analysis_column:
            st.markdown('<div class="page-kicker">SIX-CLASS PROBABILITY</div>', unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                "Material": [name.title() for name in class_names],
                "Probability": [float(value) for value in probabilities],
            }).sort_values("Probability", ascending=True)
            st.bar_chart(
                chart_data.set_index("Material"),
                horizontal=True,
                color="#dd774d",
            )
            if confidence < 0.60:
                st.warning("Low confidence. Treat this result as uncertain and check the image quality.")
            else:
                advice = WASTE_INFO[predicted_class]["advice"]
                st.markdown(
                    f'<div class="stage-note"><strong>DISPOSAL NOTE</strong><br>{advice}</div>',
                    unsafe_allow_html=True,
                )
        st.caption("Confidence is a model score, not a guarantee. Follow local sorting rules.")

else:
    st.markdown('<div class="page-kicker">02 / FEATURE INSPECTION</div>', unsafe_allow_html=True)
    st.header("Inside the network")
    st.write("Move through the feature hierarchy and compare what each stage responds to.")
    selected_stage = st.selectbox("Network stage", list(LAYER_STAGES))
    stage = LAYER_STAGES[selected_stage]
    st.markdown(architecture_markup(selected_stage), unsafe_allow_html=True)
    st.markdown(
        f'<div class="stage-note"><strong>{selected_stage}</strong> '
        f'· Output: {stage["shape"]}<br>{stage["role"]}</div>',
        unsafe_allow_html=True,
    )

    if image is None:
        st.info("Upload an image above to inspect stage activations and class-specific Grad-CAM.")
    else:
        if stage["kind"] == "spatial":
            activations = read_spatial_activations(stage["layer"], image_tensor)
            st.markdown('<div class="page-kicker">SPATIAL ACTIVATION</div>', unsafe_allow_html=True)
            channel_columns = st.columns(3)
            channel_columns[0].metric("Feature map", stage["shape"].split(" x ")[0] + " x " + stage["shape"].split(" x ")[1])
            channel_columns[1].metric("Channels", int(activations.shape[-1]))
            channel_columns[2].metric("Selected stage", stage["short"])
            channel_index = st.slider(
                "Feature channel",
                min_value=0,
                max_value=int(activations.shape[-1] - 1),
                value=0,
                key=f"channel_{stage['layer']}",
            )
            original_column, activation_column = st.columns([1, 1], gap="large")
            with original_column:
                st.markdown('<div class="page-kicker">INPUT</div>', unsafe_allow_html=True)
                st.image(image, caption="Original image", width="stretch")
            with activation_column:
                st.markdown('<div class="page-kicker">ACTIVATION MAP</div>', unsafe_allow_html=True)
                st.image(
                    activation_image(activations[:, :, channel_index]),
                    caption=f"Channel {channel_index} · normalized response",
                    width="stretch",
                )
            st.caption(
                f"Channel {channel_index} of {activations.shape[-1]} · "
                "Brighter pixels show stronger relative activation."
            )
            with st.expander("View a channel overview"):
                st.image(
                    activation_montage(activations),
                    caption="First 12 channels, independently normalized",
                    width="content",
                )
        else:
            dense_values = read_dense_activations(image_tensor)
            st.markdown('<div class="page-kicker">CLASSIFIER HEAD</div>', unsafe_allow_html=True)
            neuron_index = st.slider(
                "Dense neuron",
                min_value=0,
                max_value=int(dense_values.shape[0] - 1),
                value=0,
            )
            metric_column, chart_column = st.columns([1, 2], gap="large")
            with metric_column:
                st.metric(f"Neuron {neuron_index} activation", f"{dense_values[neuron_index]:.3f}")
                st.caption("The dense layer contains 128 learned units.")
            with chart_column:
                top_indices = np.argsort(dense_values)[-16:][::-1]
                dense_chart = pd.DataFrame({
                    "Neuron": [f"Unit {index}" for index in top_indices],
                    "Activation": [float(dense_values[index]) for index in top_indices],
                }).set_index("Neuron")
                st.bar_chart(dense_chart, horizontal=True, color="#4c8b9a")

        st.markdown('<div class="subtle-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="page-kicker">CLASS-SPECIFIC EXPLANATION</div>', unsafe_allow_html=True)
        st.header("Why this class?")
        selected_class = st.selectbox(
            "Highlight evidence for",
            class_names,
            index=predicted_index,
        )
        target_index = class_names.index(selected_class)
        with st.spinner("Tracing the selected class back to image regions..."):
            heatmap = read_gradcam(image_tensor, target_index)
        gradcam_column, legend_column = st.columns([1.15, 1], gap="large")
        with gradcam_column:
            st.image(
                gradcam_overlay(image, heatmap),
                caption=f"Warm areas contribute more to {selected_class.title()} in this view",
                width="stretch",
            )
        with legend_column:
            st.metric("Selected class", selected_class.title())
            st.write("Grad-CAM weights the final convolutional features by their influence on this class score.")
            st.caption("This is a coarse visual explanation, not a pixel-level proof of the model's reasoning.")
