import os
import streamlit as st
from PIL import Image
import numpy as np

DATASET = "classes"

soil_info = {
    "black": {"crop": "Cotton, Wheat", "fertilizer": "Urea + Potash"},
    "clay": {"crop": "Rice, Broccoli", "fertilizer": "Compost + Gypsum"},
    "red": {"crop": "Groundnut, Millets", "fertilizer": "NPK + Lime"},
    "sandy": {"crop": "Watermelon, Potato", "fertilizer": "Vermicompost"}
}

st.set_page_config(page_title="Soil AI", layout="wide")

if not os.path.exists(DATASET):
    st.error("Dataset missing ❌")
    st.stop()

# 🔥 SIMPLE FEATURE (color avg)
def get_feature(img):
    img = img.resize((100,100))
    arr = np.array(img)
    return np.mean(arr, axis=(0,1))

# 🔥 LOAD DATASET (very fast)
@st.cache_data
def load_dataset():
    features, labels = [], []
    for cls in os.listdir(DATASET):
        p = os.path.join(DATASET, cls)
        if not os.path.isdir(p): continue

        for f in os.listdir(p)[:5]:
            try:
                img = Image.open(os.path.join(p,f)).convert("RGB")
                features.append(get_feature(img))
                labels.append(cls)
            except:
                pass

    return np.array(features), np.array(labels)

features, labels = load_dataset()

# ---------------- UI ----------------
st.title("🌱 Soil Detection AI (Fast)")

file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    q = get_feature(img)

    dists = np.linalg.norm(features - q, axis=1)
    idx = np.argmin(dists)

    soil = labels[idx]
    info = soil_info[soil]

    st.success(f"Soil: {soil.upper()}")
    st.info(f"🌱 Crop: {info['crop']}")
    st.info(f"🧪 Fertilizer: {info['fertilizer']}")
