import os
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

DATASET = "clases"   # ✅ correct folder
IMG_SIZE = 224

soil_info = {
    "black": {"crop": "Cotton, Wheat", "fertilizer": "Urea + Potash"},
    "clay": {"crop": "Rice, Broccoli", "fertilizer": "Compost + Gypsum"},
    "red": {"crop": "Groundnut, Millets", "fertilizer": "NPK + Lime"},
    "sandy": {"crop": "Watermelon, Potato", "fertilizer": "Vermicompost"}
}

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Soil AI", layout="wide")

# -------- HEADER --------


# -------- MODEL --------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# -------- FEATURE --------
def extract_feature(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0)[0]
    return feat / np.linalg.norm(feat)

def extract_multi_features(img):
    w, h = img.size
    crops = [
        (0,0,w//2,h//2),
        (w//2,0,w,h//2),
        (0,h//2,w//2,h),
        (w//2,h//2,w,h),
        (w//4,h//4,3*w//4,3*h//4)
    ]
    return [extract_feature(img.crop(c)) for c in crops]

# -------- DATA --------
@st.cache_data
def load_dataset():
    features, labels = [], []
    for cls in os.listdir(DATASET):
        p = os.path.join(DATASET, cls)
        if not os.path.isdir(p): continue
        for f in os.listdir(p):
            try:
                img = Image.open(os.path.join(p,f)).convert("RGB")
                feats = extract_multi_features(img)
                for ft in feats:
                    features.append(ft)
                    labels.append(cls)
            except:
                pass
    return np.array(features), np.array(labels)

features, labels = load_dataset()

# -------- UI LAYOUT --------
# -------- PAGE CONFIG --------


# -------- CUSTOM CSS --------
st.markdown("""
<style>
.main {background: linear-gradient(120deg,#f0f9f4,#eef2ff);}
.block-container {padding-top: 1rem;}
.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}
.title {text-align:center; color:#2E8B57; font-weight:700;}
.sub {text-align:center; color:#6b7280; margin-top:-8px;}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; background:#e6f7ef; color:#1b7f5a; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("""
<h1 class="title">🌱 Soil Detection AI</h1>
<p class="sub">Smart Soil Analysis & Crop Recommendation</p>
""", unsafe_allow_html=True)

# -------- LAYOUT --------
left, right = st.columns([1.1, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Soil Image")
    file = st.file_uploader("Drop image or browse", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Preview", width=420)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Analysis Result")

    if file:
        # ---- (yahan tumhara same prediction logic rahega) ----
        query = extract_multi_features(img)
        votes = []
        for q in query:
            sims = cosine_similarity([q], features)[0]
            votes.append(labels[np.argmax(sims)])

        vals, cnt = np.unique(votes, return_counts=True)
        final = vals[np.argmax(cnt)]
        conf = (np.max(cnt)/len(votes))*100
        info = soil_info[final]

        # ---- TOP BADGE ----
        st.markdown(f"<span class='badge'>Result Ready</span>", unsafe_allow_html=True)

        # ---- METRICS (modern cards) ----
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🌍 Soil Type", final.upper())
        with c2:
            st.metric("📈 Confidence", f"{conf:.1f}%")

        # ---- DETAILS ----
        st.markdown("#### 🌱 Recommended Crops")
        st.info(info["crop"])

        st.markdown("#### 🧪 Fertilizer")
        st.info(info["fertilizer"])

        # ---- PROGRESS ----
        st.markdown("#### 📊 Confidence Level")
        st.progress(int(conf))

        # ---- ANIMATION ----
        if conf > 85:
            st.balloons()

    else:
        st.write("Upload an image to see results.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("""
<hr>
<p style='text-align:center;color:#6b7280;'>Made with ❤️ by suju</p>
""", unsafe_allow_html=True)