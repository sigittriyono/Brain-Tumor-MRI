import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
import io
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Google Drive ID kamu
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

@st.cache_resource
def load_model_from_drive():
    """Direct load ONNX dari Google Drive - NO local file!"""
    with st.spinner("📥 Loading model langsung dari Drive..."):
        # Download model ke memory
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        model_data = gdown.download(url, quiet=True, return_data=True)
        
        # Load ONNX dari bytes
        session = ort.InferenceSession(io.BytesIO(model_data))
    return session

def softmax(logits):
    """Softmax function"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def preprocess_image(image):
    """Preprocess sesuai model [1,240,240,3]"""
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.set_page_config(page_title="🧠 Brain Tumor", layout="wide")
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("**Model langsung dari Google Drive - No storage needed!**")
    
    # Load model direct dari Drive
    try:
        model = load_model_from_drive()
        st.success("✅ Model loaded dari Drive!")
        st.info(f"Input: {model.get_inputs()[0].shape}")
    except Exception as e:
        st.error(f"❌ Drive error: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("### 📊 Classes")
    st.sidebar.markdown("""
    - ✅ **No Tumor**
    - 🦠 **Glioma**  
    - 🎯 **Meningioma**
    - 🦋 **Pituitary**
    """)
    
    # Upload
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload MRI", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your MRI", use_column_width=True)
    
    if uploaded_file:
        # Process
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        with col2:
            st.subheader("🎯 Hasil Prediksi")
            
            with st.spinner("AI Analyzing..."):
                # Predict
                processed = preprocess_image(image_bgr)
                logits = model.run(None, {model.get_inputs()[0].name: processed})[0][0]
                probs = softmax(logits)
                
                # Results
                classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
                top_idx = np.argmax(probs)
                confidence = probs[top_idx] * 100
                
                # Big result
                st.markdown(f"""
                <div style="text-align:center; padding:40px; border-radius:25px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; box-shadow: 0 15px 35px rgba(0,0,0,0.2);">
                    <h1 style="font-size: 3.5em; margin: 0;">{classes[top_idx]}</h1>
                    <h2 style="font-size: 2em; margin: 15px 0 0 0; opacity: 0.95;">
                        Confidence: {confidence:.1f}%
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart
                st.subheader("📊 Confidence Scores")
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = ['#00d4aa', '#ff6b6b', '#4ecdc4', '#ffe66d']
                bars = ax.bar(classes, probs * 100, color=colors, alpha=0.8, edgecolor='black')
                
                # Highlight best
                bars[top_idx].set_alpha(1)
                bars[top_idx].set_edgecolor('darkblue')
                
                ax.set_ylabel('Confidence (%)', fontsize=14)
                ax.set_title('Brain Tumor Classification Results', fontsize=18, fontweight='bold')
                ax.set_ylim(0, 105)
                
                # Labels
                for bar, prob in zip(bars, probs*100):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, 
                           f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Table
                st.subheader("📋 Detail")
                for i, (cls, prob) in enumerate(zip(classes, probs)):
                    col1, col2 = st.columns([3,1])
                    with col1: st.write(f"**{cls}**")
                    with col2: st.metric("", f"{prob*100:.1f}%")
    
    else:
        st.info("👈 **Upload MRI scan** untuk deteksi otomatis!")
        st.markdown("**Model di-load langsung dari Google Drive - No local storage!**")

if __name__ == "__main__":
    main()
