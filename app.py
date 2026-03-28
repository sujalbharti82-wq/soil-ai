import os
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# ---------------- CONFIG ----------------
DATASET = "classes"
IMG_SIZE = 224

soil_info = {
    "black": {"crop": "Cotton, Wheat", "fertilizer": "Urea + Potash"},
    "clay": {"crop": "Rice, Broccoli", "fertilizer": "Compost + Gypsum"},
    "red": {"crop": "Groundnut, Millets", "fertilizer": "NPK + Lime"},
    "sandy": {"crop": "Watermelon, Potato", "fertilizer": "Vermicompost"}
}

st.set_page_config(page_title="Soil AI", layout="wide")

# ---------------- CHECK DATASET ----------------
if not os.path.exists(DATASET):
    st.error("❌ 'classes' folder missing (GitHub me upload karo)")
    st.stop()

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    try:
        return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    except:
        st.error("❌ Model load failed (internet/timeout issue)")
        st.stop()

model = load_model()

# ---------------- FEATURE EXTRACT ----------------
def extract_feature(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0)[0]
    return feat / np.linalg.norm(feat)

# ---------------- DATASET LOAD ----------------
@st.cache_data
def load_dataset():
    features, labels = [], []

    for cls in os.listdir(DATASET):
        path = os.path.join(DATASET, cls)
        if not os.path.isdir(path):
            continue

        files = os.listdir(path)[:5]   # 🔥 fast (limit images)

        for f in files:
            try:
                img_path = os.path.join(path, f)
                img = Image.open(img_path).convert("RGB")

                feat = extract_feature(img)
                features.append(feat)
                labels.append(cls)

            except:
                continue

    return np.array(features), np.array(labels)

# ---------------- SAFE LOAD ----------------
try:
    features, labels = load_dataset()
    if len(features) == 0:
        st.warning("⚠️ Dataset empty hai")
except:
    st.error("❌ Dataset load error")
    st.stop()

# ---------------- UI ----------------
st.markdown("""
<h1 style='text-align:center;color:#2E8B57;'>🌱 Soil Detection AI</h1>
<p style='text-align:center;color:gray;'>Smart Soil Analysis</p>
""", unsafe_allow_html=True)

left, right = st.columns([1.1,1])

# ---------------- UPLOAD ----------------
with left:
    st.subheader("📤 Upload Image")
    file = st.file_uploader("", type=["jpg","jpeg","png"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=350)

# ---------------- RESULT ----------------
with right:
    st.subheader("📊 Result")

    if file:
        if len(features) == 0:
            st.error("Dataset missing ❌")
        else:
            query = extract_feature(img)

            sims = cosine_similarity([query], features)[0]
            idx = np.argmax(sims)

            final = labels[idx]
            conf = sims[idx] * 100

            info = soil_info.get(final, {"crop": "N/A", "fertilizer": "N/A"})

            st.success(f"Soil: {final.upper()}")
            st.write(f"Confidence: {conf:.1f}%")

            st.info(f"🌱 Crop: {info['crop']}")
            st.info(f"🧪 Fertilizer: {info['fertilizer']}")

            st.progress(int(conf))

    else:
        st.write("Upload image 👆")

st.markdown("<hr><p style='text-align:center;'>Made by Suju</p>", unsafe_allow_html=True)
