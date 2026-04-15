import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_PATH = "model/brain_model.onnx"
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

@st.cache_resource
def download_model():
    from pathlib import Path
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    if not Path(MODEL_PATH).exists():
        import gdown
        with st.spinner("📥 Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    model_path = download_model()
    session = ort.InferenceSession(model_path)
    return session

def softmax(logits):
    """Proper softmax untuk multi-class"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def preprocess_image(image):
    """NHWC [1,240,240,3]"""
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_and_decode(session, image):
    """Full prediction pipeline"""
    input_meta = session.get_inputs()[0]
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Run inference
    raw_logits = session.run(None, {input_meta.name: processed})[0]
    
    # Softmax
    probabilities = softmax(raw_logits[0])
    
    return raw_logits[0], probabilities

def main():
    st.set_page_config(page_title="🧠 Brain Tumor MRI", layout="wide")
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("*High accuracy model - 98%+*")
    
    # Load model
    try:
        session = load_model()
        st.success("✅ Model ready!")
        st.info(f"Input: {session.get_inputs()[0].shape}")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    # === UPLOAD & PREDICT ===
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload MRI", type=['jpg','jpeg','png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Input MRI", use_column_width=True)
    
    if uploaded_file:
        # Process image
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        with col2:
            st.subheader("🎯 AI Prediction")
            
            # Predict
            raw_logits, probabilities = predict_and_decode(session, image_bgr)
            
            # Classes (sesuai model brain tumor standard)
            classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']  # URUTAN BENAR!
            
            top_idx = np.argmax(probabilities)
            confidence = probabilities[top_idx] * 100
            
            # === RESULT ===
            is_normal = top_idx == 0
            color = "success" if is_normal else "error"
            
            st.markdown(f"""
            <div style="text-align:center; padding:30px; border-radius:20px;
            background: linear-gradient(135deg, 
            {'#d4f4d4, #a8e6a8' if is_normal else '#f8d7da, #f5c6cb'}); 
            border:4px solid {'#28a745' if is_normal else '#dc3545'}; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                <h1 style="margin:0 0 10px 0; font-size:2.8em; 
                color:{'#28a745' if is_normal else '#dc3545'};">
                    {classes[top_idx]}
                </h1>
                <h2 style="margin:0; font-size:1.8em; opacity:0.9;">
                    Confidence: **{confidence:.1f}%**
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # === BAR CHART ===
            st.subheader("📊 Confidence Scores")
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
            
            bars = ax.bar(classes, probabilities * 100,
                         color=['#2ed573', '#ff6b6b', '#4ecdc4', '#ffa502'],
                         alpha=0.85, edgecolor='black', linewidth=2)
            
            # Highlight winner
            bars[top_idx].set_alpha(1.0)
            bars[top_idx].set_edgecolor('#000')
            bars[top_idx].set_linewidth(4)
            
            ax.set_ylabel('Confidence (%)', fontsize=14, fontweight='bold')
            ax.set_title('🧠 Brain Tumor Classification Results', 
                        fontsize=18, fontweight='bold', pad=20)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, prob) in enumerate(zip(bars, probabilities * 100)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{prob:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12)
            
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            
            # === SUMMARY TABLE ===
            st.subheader("📋 Detailed Results")
            summary_data = []
            for i, cls in enumerate(classes):
                summary_data.append({
                    'Class': cls,
                    'Confidence': f"{probabilities[i]*100:.2f}%",
                    'Score': probabilities[i]
                })
            
            st.dataframe(summary_data, use_container_width=True)
            
            # === RAW DEBUG ===
            with st.expander("🔧 Raw Debug Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Raw Logits", f"{raw_logits.max():.3f}")
                    st.metric("Max Probability", f"{probabilities.max():.3f}")
                with col2:
                    st.metric("Input Shape", str(session.get_inputs()[0].shape))
                    st.metric("Output Shape", str(raw_logits.shape))
    
    else:
        st.info("👈 **Upload MRI scan** untuk klasifikasi otomatis!")
        st.markdown("""
        ### 🎯 **Model Classes (Standard Order):**
        1. **No Tumor** ✅
        2. **Glioma** (Malignant)
        3. **Meningioma** (Benign)
        4. **Pituitary** (Benign)
        """)

if __name__ == "__main__":
    main()
