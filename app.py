import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import cv2
import matplotlib.cm as cm
import onnxruntime as ort

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="BrainMRI AI — Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# ─── Constants ───────────────────────────────────────────────
CLASSES = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]

DISEASE_INFO = {
    "glioma_tumor": (
        "Glioma Tumor",
        "Tumor otak dari sel glial, bersifat invasif dan dapat berkembang cepat."
    ),
    "meningioma_tumor": (
        "Meningioma Tumor",
        "Tumor dari meninges, umumnya jinak dan tumbuh lambat."
    ),
    "pituitary_tumor": (
        "Pituitary Tumor",
        "Tumor pada kelenjar pituitari yang mempengaruhi hormon tubuh."
    ),
    "no_tumor": (
        "No Tumor Detected",
        "Tidak ditemukan indikasi tumor pada citra MRI."
    )
}

# ─── Model ───────────────────────────────────────────────────
MODEL_PATH = "brain_model.onnx"

MODEL_PATH = "brain_model.onnx"

def download_model():
    url = "https://drive.google.com/uc?id=1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model..."):
            try:
                gdown.download(url, MODEL_PATH, quiet=False)

                if not os.path.exists(MODEL_PATH):
                    st.error("Model gagal didownload!")
                    st.stop()

            except Exception as e:
                st.error(f"Error download: {e}")
                st.stop()

download_model()

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

session = load_model()

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

session = load_model()

# ─── Saliency (Occlusion) ────────────────────────────────────
def compute_saliency(session, img_array):
    input_name = session.get_inputs()[0].name
    base_pred = session.run(None, {input_name: img_array})[0][0]
    pred_class = int(np.argmax(base_pred))
    base_score = base_pred[pred_class]

    h, w = 240, 240
    saliency = np.zeros((h, w))

    for y in range(0, h, 28):
        for x in range(0, w, 28):
            occluded = img_array.copy()
            occluded[0, y:y+56, x:x+56, :] = 0
            pred = session.run(None, {input_name: occluded})[0][0]
            saliency[y:y+56, x:x+56] += (base_score - pred[pred_class])

    saliency = np.maximum(saliency, 0)
    saliency /= (saliency.max() + 1e-8)
    saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
    return saliency

def make_overlay(img, saliency):
    img = np.array(img.resize((240,240)))
    heatmap = (cm.jet(saliency)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

# ─── Validasi MRI ────────────────────────────────────────────
def is_valid_mri(probs, image):
    max_conf = np.max(probs)

    if max_conf < 0.5:
        return False, "Confidence terlalu rendah"

    gray = np.array(image.convert("L"))
    brightness = gray.mean()

    if brightness < 30:
        return False, "Gambar terlalu gelap"
    if brightness > 220:
        return False, "Gambar terlalu terang"

    return True, "OK"

# ─── UI Header ───────────────────────────────────────────────
st.title("🧠 BrainMRI AI")
st.caption("Klasifikasi Tumor Otak dari Citra MRI menggunakan Deep Learning")

# ─── Layout ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

# LEFT
with col1:
    uploaded_file = st.file_uploader("Upload Citra MRI", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview", use_container_width=True)

        # info tambahan
        w, h = image.size
        gray = np.array(image.convert("L"))
        brightness = gray.mean()

        st.write(f"Dimensi: {w}x{h}")
        st.write(f"Brightness: {brightness:.2f}")

# RIGHT
with col2:
    if uploaded_file:
        # preprocessing
        img = image.resize((240,240))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # inference
        input_name = session.get_inputs()[0].name
        probs = session.run(None, {input_name: img_array})[0][0]

        # validasi
        valid, reason = is_valid_mri(probs, image)

        if not valid:
            st.error(f"Gambar tidak valid: {reason}")
            st.stop()

        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]

        full_name, desc = DISEASE_INFO[pred_class]

        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** {full_name}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.info(desc)

        # Probabilities
        st.subheader("Distribusi Probabilitas")
        for i, cls in enumerate(CLASSES):
            st.write(f"{cls}: {probs[i]*100:.2f}%")

        # Saliency
        st.subheader("Grad-CAM (Occlusion)")
        saliency = compute_saliency(session, img_array)
        overlay = make_overlay(image, saliency)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(image.resize((240,240)), caption="Original")
        with c2:
            st.image((cm.jet(saliency)[:, :, :3]), caption="Heatmap")
        with c3:
            st.image(overlay, caption="Overlay")
