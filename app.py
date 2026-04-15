import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_PATH = "model/brain_model.onnx"
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"  # ID kamu

def download_model():
    """Download besar dari Google Drive - NO SIZE LIMIT!"""
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    if not Path(MODEL_PATH).exists():
        with st.spinner("📥 Downloading large model... Ini bisa 1-2 menit"):
            # Direct download link
            url = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded!")
    return MODEL_PATH

@st.cache_resource
def load_model():
    model_path = download_model()
    session = ort.InferenceSession(model_path)
    return session

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def preprocess_image(image):
    """Sesuai model kamu [1,240,240,3]"""
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.set_page_config(page_title="🧠 Brain Tumor", layout="wide")
    st.title("🧠 Brain Tumor MRI Classifier")
    
    # Load model (auto download)
    try:
        with st.spinner("Loading AI model..."):
            session = load_model()
        st.success("✅ Model ready!")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    # Upload
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload MRI", type=['jpg','png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        st.image(image, caption="Your MRI", use_column_width=True)
        
        with col2:
            # Predict
            processed = preprocess_image(image_bgr)
            logits = session.run(None, {session.get_inputs()[0].name: processed})[0][0]
            probs = softmax(logits)
            
            classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
            top_idx = np.argmax(probs)
            conf = probs[top_idx] * 100
            
            # Result
            st.markdown(f"""
            <div style="text-align:center;padding:30px;border-radius:20px;
            background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            color:white;box-shadow:0 10px 30px rgba(0,0,0,0.3);">
                <h1 style="margin:0;font-size:3em;">{classes[top_idx]}</h1>
                <h2 style="margin:10px 0 0 0;">{conf:.1f}% Confidence</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10,6))
            colors = ['green' if i==top_idx else 'lightblue' for i in range(4)]
            bars = ax.bar(classes, probs*100, color=colors, alpha=0.8)
            ax.set_title("Confidence Scores", fontsize=16)
            ax.set_ylabel("Probability %")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.info("👈 Upload MRI untuk hasil instan!")

if __name__ == "__main__":
    main()
