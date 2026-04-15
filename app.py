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

CLASS_DESC = {
    "Glioma Tumor": (
        "Glioma adalah tumor yang berasal dari sel glial di otak atau tulang belakang. "
        "Termasuk salah satu jenis tumor otak yang paling umum dan bisa bersifat jinak "
        "hingga sangat agresif tergantung gradenya."
    ),
    "Meningioma Tumor": (
        "Meningioma adalah tumor yang tumbuh dari meninges (selaput pelindung otak dan "
        "sumsum tulang belakang). Umumnya bersifat jinak dan tumbuh lambat, namun dapat "
        "menekan jaringan otak di sekitarnya."
    ),
    "No Tumor": (
        "Tidak ditemukan indikasi tumor pada citra MRI ini. Struktur otak tampak normal. "
        "Tetap konsultasikan dengan dokter spesialis untuk diagnosis resmi."
    ),
    "Pituitary Tumor": (
        "Tumor hipofisis (pituitary) adalah tumor yang tumbuh di kelenjar pituitari di "
        "dasar otak. Sebagian besar bersifat jinak (adenoma) dan dapat mempengaruhi "
        "produksi hormon tubuh."
    ),
}

# EfficientNet mean & std dari ImageNet (sama persis dengan tf.keras preprocess_input)
EFFICIENTNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EFFICIENTNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

    session   = ort.InferenceSession(MODEL_PATH)
    inp       = session.get_inputs()[0]
    inp_shape = inp.shape   # contoh: [1, 240, 240, 3] atau [None, 240, 240, 3]

    # Deteksi apakah channels-first (PyTorch style) atau channels-last (TF/Keras style)
    channels_first = (len(inp_shape) == 4 and inp_shape[1] == 3)

    return session, inp_shape, channels_first

# ============================================================
# PREPROCESSING
# ============================================================
#
# tf.keras EfficientNetB1 preprocess_input:
#   pixel [0,255] → /255.0 → subtract mean → bagi std  (range ~[-2.1, 2.6])
#
# Ini BERBEDA dari:
#   MobileNet  : x / 127.5 - 1       (range [-1, 1])
#   ResNet/VGG : subtract mean BGR saja, tidak bagi std
#
# Fungsi di bawah mencoba 2 mode lalu memilih yang menghasilkan
# prediksi paling "yakin" (entropy terendah).
# ============================================================

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def _entropy(probs):
    """Shannon entropy — makin rendah = model makin yakin."""
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

def _run_inference(session, img_batch):
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: img_batch})
    probs      = outputs[0][0].astype(np.float32)
    # Terapkan softmax jika output masih logits (belum di-softmax)
    if probs.min() < 0 or probs.max() > 1.0 or abs(probs.sum() - 1.0) > 0.05:
        probs = _softmax(probs)
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

def _make_batch(img_array, channels_first):
    if channels_first:
        img_array = np.transpose(img_array, (2, 0, 1))   # HWC → CHW
    return np.expand_dims(img_array, axis=0)

def predict_with_autodetect(session, img_pil, channels_first):
    """
    Coba 2 mode preprocessing EfficientNet, pilih yang entropy-nya lebih rendah.
    Mode A: EfficientNet standar  → /255 → subtract mean → bagi std
    Mode B: Raw [0,255]           → untuk ONNX yg sudah include preprocessing layer
    """
    img_resized = img_pil.resize(IMG_SIZE)
    img_array   = np.array(img_resized, dtype=np.float32)   # (240,240,3) range [0,255]

    # ── Mode A: EfficientNet normalize ──────────────────────
    img_A = (img_array / 255.0 - EFFICIENTNET_MEAN) / EFFICIENTNET_STD
    batch_A = _make_batch(img_A, channels_first)
    probs_A, pred_A = _run_inference(session, batch_A)
    ent_A = _entropy(probs_A)

    # ── Mode B: Raw pixels [0,255] ───────────────────────────
    batch_B = _make_batch(img_array, channels_first)
    probs_B, pred_B = _run_inference(session, batch_B)
    ent_B = _entropy(probs_B)

    if ent_A <= ent_B:
        return probs_A, pred_A, img_resized, img_A, "EfficientNet (/255 → mean/std ImageNet)"
    else:
        return probs_B, pred_B, img_resized, img_array, "Raw pixels (preprocessing ada di model)"

# ============================================================
# VISUALISASI PREPROCESSING
# ============================================================

def plot_preprocessing(img_pil, img_resized, img_preprocessed, mode_used):
    orig_w, orig_h = img_pil.size
    img_gray = np.mean(np.array(img_pil.convert("RGB")), axis=2).astype(np.uint8)

    # Normalisasi ke [0,1] untuk display
    disp = img_preprocessed.copy()
    if disp.ndim == 3 and disp.shape[0] == 3:
        disp = np.transpose(disp, (1, 2, 0))
    disp = disp - disp.min()
    disp = disp / (disp.max() + 1e-8)

    pct  = int(240 / orig_w * 100)
    vmin = float(img_preprocessed.min())
    vmax = float(img_preprocessed.max())

    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(1, 9, figure=fig,
                            width_ratios=[3, 0.3, 3, 0.3, 3, 0.3, 3, 0.3, 3])

    steps = [
        (0, np.array(img_pil.convert("RGB")), None,
         f"{orig_w}×{orig_h}×3", "Gambar Asli", "Input dari user", "#495057"),
        (2, img_gray, "gray",
         f"{orig_w}×{orig_h}×1", "Grayscale", "Konversi ke 1 channel", "#495057"),
        (4, np.array(img_resized), None,
         "240×240×3", "Resize", f"target_size=(240,240)\n({pct}% dari asli)", "#1971C2"),
        (6, disp, None,
         f"min={vmin:.2f}  max={vmax:.2f}",
         "preprocess_input", "EfficientNetB1\n/255 → mean/std ImageNet", "#0F6E56"),
        (8, disp, None,
         "(1, 240, 240, 3)", "expand_dims", "Siap masuk model", "#5F3DC4"),
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
        if ax_idx < 8:
            ax.annotate("", xy=(1.12, 0.5), xycoords="axes fraction",
                        xytext=(1.02, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#ADB5BD", lw=1.5))

    plt.suptitle(
        f"Pipeline Preprocessing MRI — EfficientNetB1\n[Mode terpakai: {mode_used}]",
        fontsize=11, fontweight="bold", y=1.06)
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
    page_title="Klasifikasi Tumor Otak Berbasis Citra MRI",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<h1 style='text-align:center; color:#2C3E50;'>🧠 Klasifikasi Tumor Otak Berbasis Citra MRI</h1>
<p style='text-align:center; color:#7F8C8D; font-size:16px;'>
    Menggunakan Metode EfficientNetB1 (Transfer Learning)
</p>
<hr style='border:0.5px solid #E0E0E0; margin-bottom:2rem'>
""", unsafe_allow_html=True)

try:
    session, inp_shape, channels_first = load_model()
    st.success(f"✅ Model berhasil dimuat! Input shape: {inp_shape}", icon="✅")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

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
    st.markdown(f"""
    - **Arsitektur** : EfficientNetB1
    - **Input size** : 240 × 240 × 3
    - **Pretrained** : ImageNet
    - **Fine-tune**  : Top 60 layers
    - **Accuracy**   : ~83%
    - **Input shape** : `{inp_shape}`
    """)
    st.markdown("---")
    st.markdown("### 📁 Format Gambar")
    st.markdown("JPG, JPEG, atau PNG")

st.markdown("### 📂 Upload Gambar MRI")
uploaded_file = st.file_uploader(
    "Pilih gambar MRI otak",
    type=["jpg", "jpeg", "png"],
    help="Upload gambar MRI otak untuk diklasifikasikan",
)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Menganalisis gambar..."):
        probs, pred_idx, img_resized, img_preprocessed, mode_used = \
            predict_with_autodetect(session, img_pil, channels_first)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    pred_color = CLASS_COLORS[pred_idx]
    is_tumor   = pred_class != "No Tumor"

    st.info(f"🔧 Preprocessing mode: **{mode_used}**", icon="ℹ️")
    st.markdown("---")
    st.markdown("### 🎯 Hasil Prediksi")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(img_pil,
                 caption=f"Gambar MRI ({img_pil.width}×{img_pil.height}px)",
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

    st.markdown("---")
    st.markdown("### ⚙️ Visualisasi Pipeline Preprocessing")
    st.caption("Berikut tahapan transformasi gambar sebelum masuk ke model:")

    fig_prep = plot_preprocessing(img_pil, img_resized, img_preprocessed, mode_used)
    st.pyplot(fig_prep, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("### 📊 Detail Probabilitas per Kelas")

    cols = st.columns(4)
    for i, (cls, prob, color) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLORS)):
        with cols[i]:
            is_pred = i == pred_idx
            border  = f"3px solid {color}" if is_pred else "0.5px solid #E0E0E0"
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

st.markdown("""
<hr style='border:0.5px solid #E0E0E0; margin-top:3rem'>
<p style='text-align:center; color:#ADB5BD; font-size:12px'>
    Klasifikasi Tumor Otak Berbasis Citra MRI Menggunakan Metode EfficientNetB1 |
    Tugas Machine Learning Bioinformatika
</p>
""", unsafe_allow_html=True)
