import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import gdown
import os
import onnxruntime as ort
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH   = "model/brain_model.onnx"
GDRIVE_ID    = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"
CLASS_NAMES  = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
IMG_SIZE     = (240, 240)
CLASS_COLORS = ["#E74C3C", "#E67E22", "#27AE60", "#2980B9"]

# ============================================================
# LOAD MODEL (ONNX)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        with st.spinner("⏬ Mengunduh model ONNX..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess_image(img_pil):
    img_resized = img_pil.resize(IMG_SIZE)
    img_array   = np.array(img_resized).astype(np.float32)
    img_array = (img_array / 127.5) - 1   # recommended EfficientNet
    img_batch = np.expand_dims(img_array, axis=0)

    return img_resized, img_array, img_batch

def predict(session, img_batch):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_batch})
    probs = outputs[0][0]
    pred_idx = np.argmax(probs)
    return probs, pred_idx

# ============================================================
# VISUALISASI PREPROCESSING
# ============================================================

def plot_preprocessing(img_pil, img_resized, img_array, img_preprocessed):
    orig_w, orig_h = img_pil.size

    img_gray = np.mean(np.array(img_pil.convert("RGB")), axis=2).astype(np.uint8)

    disp_prep = img_preprocessed - img_preprocessed.min()
    disp_prep = disp_prep / disp_prep.max()

    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(1, 9, figure=fig,
                            width_ratios=[3, 0.3, 3, 0.3, 3, 0.3, 3, 0.3, 3])

    steps = [
        (0,  np.array(img_pil.convert("RGB")), None,   f"{orig_w}×{orig_h}×3",
         "Gambar Asli",       "Input dari user",                  "#495057"),
        (2,  img_gray,                          "gray", f"{orig_w}×{orig_h}×1",
         "Grayscale",         "Konversi ke 1 channel",             "#495057"),
        (4,  np.array(img_resized),             None,   "240×240×3",
         "Resize",            f"target_size=(240,240)\n({int(240/orig_w*100)}% dari asli)", "#1971C2"),
        (6,  disp_prep,                         None,
         f"min={img_preprocessed.min():.2f}\nmax={img_preprocessed.max():.2f}",
         "preprocess_input",  "EfficientNet normalize\nrange: [-1, 1]",            "#0F6E56"),
        (8,  disp_prep,                         None,   "(1, 240, 240, 3)",
         "expand_dims",       "Siap masuk model",                  "#5F3DC4"),
    ]

    for ax_idx, img, cmap, badge, title, sub, color in steps:
        ax = fig.add_subplot(gs[ax_idx])
        if title == "expand_dims":
            ax.set_facecolor("#EDE9FE")
            ax.text(0.5, 0.55, "(1, 240, 240, 3)", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#3C3489",
                    transform=ax.transAxes)
            ax.text(0.5, 0.38, "(batch, H, W, channel)", ha="center", va="center",
                    fontsize=9, color="#5F3DC4", transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.5); spine.set_visible(True)
        ax.set_title(badge, fontsize=9, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"), pad=6)
        ax.set_xlabel(f"{title}\n{sub}", fontsize=9, labelpad=6, color="#212529")

        # Panah ke step berikutnya
        if ax_idx < 8:
            ax.annotate("", xy=(1.12, 0.5), xycoords="axes fraction",
                        xytext=(1.02, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#ADB5BD", lw=1.5))

    plt.suptitle("Pipeline Preprocessing MRI — EfficientNetB1",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout(w_pad=1.5)
    return fig

# ============================================================
# VISUALISASI CONFIDENCE BAR
# ============================================================

def plot_confidence(probs, pred_idx):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#F8F9FA")

    short = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    bars  = ax.barh(short, probs * 100, color=CLASS_COLORS,
                    edgecolor="white", linewidth=0.8, height=0.55)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold", color="#2C3E50")

    bars[pred_idx].set_edgecolor(CLASS_COLORS[pred_idx])
    bars[pred_idx].set_linewidth(2.5)

    ax.set_xlim(0, 115)
    ax.set_xlabel("Probabilitas (%)", fontsize=10)
    ax.set_title("Distribusi Probabilitas Prediksi", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("white")
    plt.tight_layout()
    return fig

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

# Header
st.markdown("""
<h1 style='text-align:center; color:#2C3E50;'>🧠 Brain Tumor MRI Classifier</h1>
<p style='text-align:center; color:#7F8C8D; font-size:16px;'>
    Klasifikasi jenis tumor otak menggunakan EfficientNetB1 (Transfer Learning)
</p>
<hr style='border:0.5px solid #E0E0E0; margin-bottom:2rem'>
""", unsafe_allow_html=True)

# Load model
try:
    session = load_model()
    st.success("✅ Model berhasil dimuat!", icon="✅")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini mengklasifikasikan gambar MRI otak ke dalam **4 kelas**:
    - 🔴 Glioma Tumor
    - 🟠 Meningioma Tumor
    - 🟢 No Tumor
    - 🔵 Pituitary Tumor
    """)
    st.markdown("---")
    st.markdown("### 📊 Info Model")
    st.markdown("""
    - **Arsitektur** : EfficientNetB1
    - **Input size** : 240 × 240 × 3
    - **Pretrained** : ImageNet
    - **Fine-tune**  : Top 60 layers
    - **Accuracy**   : ~83%
    """)
    st.markdown("---")
    st.markdown("### 📁 Format Gambar")
    st.markdown("JPG, JPEG, atau PNG")

# Upload gambar
st.markdown("### 📂 Upload Gambar MRI")
uploaded_file = st.file_uploader(
    "Pilih gambar MRI otak",
    type=["jpg", "jpeg", "png"],
    help="Upload gambar MRI otak untuk diklasifikasikan",
)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")

    # Preprocessing
    img_resized, img_array, img_batch = preprocess_image(img_pil)

    # Prediksi
    with st.spinner("🔍 Menganalisis gambar..."):
        probs, pred_idx = predict(session, img_batch)

    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx] * 100
    pred_color  = CLASS_COLORS[pred_idx]
    is_tumor    = pred_class != "No Tumor"

    st.markdown("---")

    # ── Hasil Prediksi ───────────────────────────────────────
    st.markdown("### 🎯 Hasil Prediksi")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(img_pil, caption=f"Gambar MRI ({img_pil.width}×{img_pil.height}px)",
                 use_container_width=True)

    with col2:
        status_icon = "⚠️" if is_tumor else "✅"
        status_text = "TUMOR TERDETEKSI" if is_tumor else "TIDAK ADA TUMOR"
        st.markdown(f"""
        <div style='background-color:#F8F9FA; padding:20px; border-radius:12px;
                    border-left:5px solid {pred_color}; margin-bottom:16px'>
            <h3 style='color:{pred_color}; margin:0'>{status_icon} {status_text}</h3>
            <h2 style='color:#2C3E50; margin:8px 0'>{pred_class}</h2>
            <p style='font-size:28px; font-weight:bold; color:{pred_color}; margin:0'>
                {confidence:.1f}%
            </p>
            <p style='color:#7F8C8D; margin:4px 0'>Confidence Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#F0F4FF; padding:14px; border-radius:8px;
                    border:0.5px solid #D0D8F0'>
            <p style='color:#444; margin:0; font-size:14px; line-height:1.6'>
                📌 <b>Deskripsi:</b><br>{CLASS_DESC[pred_class]}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        fig_conf = plot_confidence(probs, pred_idx)
        st.pyplot(fig_conf, use_container_width=True)
        plt.close()

    # ── Visualisasi Preprocessing ───────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Visualisasi Pipeline Preprocessing")
    st.caption("Berikut tahapan transformasi gambar sebelum masuk ke model:")

    fig_prep = plot_preprocessing(img_pil, img_resized, img_array, img_preprocessed)
    st.pyplot(fig_prep, use_container_width=True)
    plt.close()

    # ── Detail probabilitas ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Detail Probabilitas per Kelas")

    cols = st.columns(4)
    for i, (cls, prob, color) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLORS)):
        with cols[i]:
            is_pred = i == pred_idx
            border  = f"3px solid {color}" if is_pred else f"0.5px solid #E0E0E0"
            st.markdown(f"""
            <div style='text-align:center; padding:16px; border-radius:10px;
                        border:{border}; background:{"#F8F9FA" if is_pred else "white"}'>
                <p style='font-size:13px; color:#7F8C8D; margin:0'>{cls}</p>
                <p style='font-size:26px; font-weight:bold; color:{color}; margin:4px 0'>
                    {prob*100:.1f}%
                </p>
                {"<p style='font-size:11px; color:" + color + "; margin:0'>▲ Prediksi</p>" if is_pred else ""}
            </div>
            """, unsafe_allow_html=True)

else:
    # Placeholder saat belum upload
    st.markdown("""
    <div style='text-align:center; padding:60px; background:#F8F9FA;
                border-radius:12px; border:2px dashed #D0D0D0; margin-top:1rem'>
        <p style='font-size:48px; margin:0'>🧠</p>
        <p style='font-size:18px; color:#7F8C8D; margin:8px 0'>
            Upload gambar MRI otak untuk memulai klasifikasi
        </p>
        <p style='font-size:13px; color:#ADB5BD'>
            Format yang didukung: JPG, JPEG, PNG
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style='border:0.5px solid #E0E0E0; margin-top:3rem'>
<p style='text-align:center; color:#ADB5BD; font-size:12px'>
    Brain Tumor MRI Classifier — EfficientNetB1 Transfer Learning |
    Tugas Machine Learning Bioinformatika
</p>
""", unsafe_allow_html=True)
