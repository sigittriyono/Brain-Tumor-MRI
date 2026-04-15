import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from PIL import Image
from pathlib import Path

MODEL_PATH = "model/brain_model.onnx"
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

@st.cache_resource
def download_model():
    """Download model jika belum ada"""
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    if not Path(MODEL_PATH).exists():
        with st.spinner("📥 Downloading model dari Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    """Load ONNX model"""
    model_path = download_model()
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

def preprocess_image(image):
    """Format: [1, 240, 240, 3] NHWC"""
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # [1, 240, 240, 3]
    return image

def predict(session, image):
    """Prediksi dengan debug info"""
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    
    processed = preprocess_image(image)
    
    st.info(f"**Model input**: {input_shape}")
    st.info(f"**Processed**: {processed.shape}")
    
    predictions = session.run(None, {input_name: processed})[0]
    return predictions

def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier", 
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("Upload MRI scan untuk deteksi tumor otak")
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            session = load_model()
        input_shape = session.get_inputs()[0].shape
        st.success(f"✅ Model siap! Input shape: {input_shape}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Info")
        st.markdown("""
        **Kelas:**
        - 🦠 **Glioma** - Tumor ganas
        - 🎯 **Meningioma** - Tumor selaput otak  
        - ✅ **No Tumor** - Normal
        - 🦋 **Pituitary** - Tumor kelenjar
        
        **Input**: 240x240 pixels
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload MRI Image", 
            type=['jpg', 'jpeg', 'png', 'JPG', 'PNG'],
            help="Upload scan MRI otak"
        )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # BGR conversion
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR) if image_array.shape[2] == 4 else image_array
        
        st.subheader("📸 Gambar Asli")
        st.image(image, use_column_width=True)
        
        # Prediction
        with col2:
            st.subheader("🎯 Hasil Prediksi")
            
            with st.spinner("🔍 Menganalisis..."):
                predictions = predict(session, image_bgr)
                scores = np.softmax(predictions[0])
                
                # Classes
                classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                top_idx = np.argmax(scores)
                confidence = scores[top_idx] * 100
                
                # Main result
                result_emoji = "✅" if top_idx == 2 else "⚠️"
                color = "green" if top_idx == 2 else "red"
                
                st.markdown(f"""
                <div style="text-align:center; padding:30px; margin:20px 0;
                border-radius:15px; border:4px solid {color};
                background: linear-gradient(145deg, {'#d4f4d4' if top_idx==2 else '#ffd4d4'}, white);">
                    <h1 style="color:{color}; margin:0;">{result_emoji} {classes[top_idx]}</h1>
                    <h2 style="color:{color}; margin:10px 0 0 0;">{confidence:.1f}% Confidence</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart
                st.subheader("📊 Confidence Semua Kelas")
                chart_data = dict(zip(classes, scores * 100))
                st.bar_chart(chart_data, height=400)
                
                # Detail
                st.subheader("📋 Detail Prediksi")
                for i, (cls, score) in enumerate(zip(classes, scores)):
                    col_a, col_b = st.columns([3,1])
                    with col_a:
                        st.write(f"**{cls}**")
                    with col_b:
                        st.metric("", f"{score*100:.1f}%")
                
                # Debug
                with st.expander("🔧 Debug Info"):
                    st.json({
                        "Model input": str(session.get_inputs()[0].shape),
                        "Predictions shape": str(predictions.shape),
                        "Top prediction": f"{classes[top_idx]} ({confidence:.2f}%)"
                    })
    
    else:
        st.info("👈 **Upload file MRI** di sebelah kiri untuk mulai!")
        
        # Example images
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.image(
                "https://images.unsplash.com/photo-1588735374723-4f0515d4dcbe?w=300", 
                caption="Contoh: Normal"
            )
        with col_ex2:
            st.image(
                "https://images.unsplash.com/photo-1629429843630-b7c6041f5906?w=300", 
                caption="Contoh: Tumor"
            )

if __name__ == "__main__":
    main()
