import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = "model/brain_model.onnx"
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

@st.cache_resource
def download_model():
    """Download model dari Google Drive jika belum ada"""
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    if not Path(MODEL_PATH).exists():
        with st.spinner("Downloading model dari Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    """Load ONNX model"""
    model_path = download_model()
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image, img_size=(224, 224)):
    """Preprocess gambar untuk model"""
    # Resize
    image = cv2.resize(image, img_size)
    # Normalize ke 0-1
    image = image.astype(np.float32) / 255.0
    # Tambah dimension batch
    image = np.expand_dims(image, axis=0)
    # Transpose ke CHW format (ONNX biasanya)
    image = np.transpose(image, (0, 3, 1, 2))
    return image

def predict(session, image):
    """Lakukan prediksi"""
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    processed_image = preprocess_image(image)
    
    # Pastikan shape sesuai
    if len(processed_image.shape) == 4:
        predictions = session.run(None, {input_name: processed_image.astype(np.float32)})[0]
    else:
        predictions = session.run(None, {input_name: processed_image[None].astype(np.float32)})[0]
    
    return predictions

def main():
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("---")
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            session = load_model()
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📁 Upload Gambar")
    uploaded_file = st.sidebar.file_uploader(
        "Pilih file MRI (.jpg, .jpeg, .png)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar MRI otak"
    )
    
    st.sidebar.header("ℹ️ Informasi")
    st.sidebar.info("""
    **Kelas Tumor:**
    - **Glioma**: Tumor otak ganas
    - **Meningioma**: Tumor selaput otak
    - **Pituitary**: Tumor kelenjar pituitari
    - **No Tumor**: Tidak ada tumor
    
    **Akurasi Model**: 98%+
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Tampilkan gambar asli
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with col1:
            st.subheader("📸 Gambar MRI")
            st.image(image, use_column_width=True)
        
        # Prediksi
        with st.spinner("🔄 Memproses..."):
            predictions = predict(session, image_cv)
            scores = np.softmax(predictions[0])
            
            # Class names
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            
            # Hasil prediksi
            top_idx = np.argmax(scores)
            confidence = scores[top_idx] * 100
            
            with col2:
                st.subheader("🎯 Hasil Prediksi")
                
                # Badge hasil
                if top_idx == 2:  # No Tumor
                    st.success(f"**{class_names[top_idx]}**")
                else:
                    st.error(f"**{class_names[top_idx]}**")
                
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Bar chart
                st.subheader("📊 Confidence Scores")
                chart_data = dict(zip(class_names, scores * 100))
                st.bar_chart(chart_data)
                
                # Detail semua kelas
                st.subheader("📋 Detail Semua Kelas")
                for i, (cls, score) in enumerate(zip(class_names, scores)):
                    st.write(f"**{cls}**: {score*100:.1f}%")
    
    else:
        # Placeholder gambar contoh
        st.info("👈 Silahkan upload gambar MRI di sidebar untuk memulai!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://images.unsplash.com/photo-1588735374723-4f0515d4dcbe?w=400", 
                    caption="Contoh MRI Normal")
        with col2:
            st.image("https://images.unsplash.com/photo-1629429843630-b7c6041f5906?w=400", 
                    caption="Contoh MRI Tumor")

if __name__ == "__main__":
    main()
