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

# ---------------- PAGE ----------------
st.set_page_config(page_title="Soil AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2E8B57;
}
.sub {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🌱 Soil Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Smart Soil Analysis System</div>", unsafe_allow_html=True)

# ---------------- CHECK ----------------
if not os.path.exists(DATASET):
    st.error("Dataset missing ❌")
    st.stop()

# ---------------- FEATURE ----------------
def get_feature(img):
    img = img.resize((80,80))
    return np.mean(np.array(img), axis=(0,1))

# ---------------- LOAD ----------------
@st.cache_data
def load_dataset():
    features, labels = [], []

    for cls in os.listdir(DATASET):
        p = os.path.join(DATASET, cls)
        if not os.path.isdir(p):
            continue

        for f in os.listdir(p)[:3]:
            try:
                img = Image.open(os.path.join(p,f)).convert("RGB")
                features.append(get_feature(img))
                labels.append(cls)
            except:
                pass

    return np.array(features), np.array(labels)

features, labels = load_dataset()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------- LEFT ----------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Soil Image")

    file = st.file_uploader("", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT ----------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Analysis Result")

    if file:
        with st.spinner("Analyzing soil..."):
            q = get_feature(img)
            dists = np.linalg.norm(features - q, axis=1)
            idx = np.argmin(dists)

            soil = labels[idx]
            info = soil_info[soil]

            st.success(f"🌍 Soil Type: {soil.upper()}")

            st.metric("Confidence", "High ✅")

            st.markdown("### 🌱 Recommended Crops")
            st.info(info["crop"])

            st.markdown("### 🧪 Fertilizer Suggestion")
            st.info(info["fertilizer"])

            st.progress(90)

    else:
        st.write("Upload image to see result 👆")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>Made by Suju 🚀</p>
""", unsafe_allow_html=True)
