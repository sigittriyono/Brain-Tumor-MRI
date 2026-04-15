import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import requests
import io
from PIL import Image
import matplotlib.pyplot as plt

# Google Drive ID kamu
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

def download_drive_file(gdrive_id):
    """Download ONNX direct dari Drive ke memory"""
    URL = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
    
    session = requests.Session()
    response = session.get(URL)
    
    # Handle large file confirmation
    if 'content-disposition' not in response.headers:
        # Get confirm token
        for key, value in response.cookies.items():
            if 'download_warning' in key:
                confirm_token = value
                break
        
        params = {'id': gdrive_id, 'confirm': confirm_token}
        response = session.get(URL, params=params, stream=True)
    
    return response.content

@st.cache_resource
def load_model_from_drive():
    """Load ONNX langsung dari Drive bytes"""
    with st.spinner("📥 Loading model dari Google Drive..."):
        model_bytes = download_drive_file(GDRIVE_ID)
        model_stream = io.BytesIO(model_bytes)
        session = ort.InferenceSession(model_stream)
    st.success("✅ Model loaded dari Drive!")
    return session

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def preprocess_image(image):
    """[1, 240, 240, 3] NHWC untuk model kamu"""
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.set_page_config(page_title="🧠 Brain Tumor", layout="wide")
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("*Direct Google Drive model loading*")
    
    # Load model
    try:
        model = load_model_from_drive()
        input_shape = model.get_inputs()[0].shape
        st.success(f"✅ Ready! Shape: {input_shape}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()
    
    # Upload
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload MRI", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="MRI Scan", use_column_width=True)
    
    if uploaded_file:
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        with col2:
            st.subheader("🎯 Prediction")
            with st.spinner("Analyzing..."):
                processed = preprocess_image(image_bgr)
                logits = model.run(None, {model.get_inputs()[0].name: processed})[0][0]
                probs = softmax(logits)
                
                classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
                top_idx = np.argmax(probs)
                confidence = probs[top_idx] * 100
                
                # Result card
                st.markdown(f"""
                <div style="text-align:center;padding:35px;border-radius:20px;
                background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                color:white;box-shadow:0 10px 30px rgba(0,0,0,0.3);">
                    <h1 style="font-size:3em;margin:0;">{classes[top_idx]}</h1>
                    <h2 style="font-size:2em;margin:15px 0 0 0;">{confidence:.1f}% Confidence</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart
                st.subheader("📊 Confidence Scores")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#00d4aa', '#ff6b6b', '#4ecdc4', '#ffe66d']
                bars = ax.bar(classes, probs*100, color=colors, alpha=0.8)
                bars[top_idx].set_alpha(1.0)
                
                ax.set_ylabel('Confidence (%)')
                ax.set_title('Brain Tumor Classification')
                ax.set_ylim(0, 105)
                
                # Add labels
                for bar, prob in zip(bars, probs*100):
                    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+1,
                           f'{prob:.1f}%', ha='center', fontweight='bold')
                
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Debug
                with st.expander("🔧 Debug"):
                    st.json({
                        "Input shape": str(model.get_inputs()[0].shape),
                        "Probs": {classes[i]: f"{p*100:.2f}%" for i,p in enumerate(probs)},
                        "Logits max": f"{logits.max():.3f}"
                    })
    
    else:
        st.info("👈 Upload MRI untuk klasifikasi!")

if __name__ == "__main__":
    main()
