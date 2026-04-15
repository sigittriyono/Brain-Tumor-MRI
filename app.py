import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from PIL import Image
import os
from pathlib import Path

MODEL_PATH = "model/brain_model.onnx"
GDRIVE_ID = "1-dCqvMmQAoxuvTte-fGLEu4Jbyzs9iYH"

@st.cache_resource
def download_model():
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    if not Path(MODEL_PATH).exists():
        with st.spinner("📥 Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    model_path = download_model()
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

def preprocess_image(image, target_size):
    """Preprocess dengan exact model shape"""
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def predict(session, image):
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    
    # Extract H,W dari [batch, channels, H, W]
    target_size = (input_shape[2], input_shape[3])
    
    processed_image = preprocess_image(image, target_size)
    
    predictions = session.run(None, {input_name: processed_image.astype(np.float32)})[0]
    return predictions, input_shape

def main():
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("**Fixed input shape untuk model 240x240**")
    
    try:
        session = load_model()
        input_shape = session.get_inputs()[0].shape
        st.success(f"✅ Model loaded! Input: {input_shape}")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    # Sidebar
    uploaded_file = st.sidebar.file_uploader("Upload MRI", type=['jpg','jpeg','png'])
    
    col1, col2 = st.columns([1,1])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(image, use_column_width=True)
        
        with col2:
            with st.spinner("🔍 Predicting..."):
                predictions, model_shape = predict(session, image_cv)
                scores = np.softmax(predictions[0])
                
                class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                top_idx = np.argmax(scores)
                confidence = scores[top_idx] * 100
                
                st.subheader("🎯 Result")
                result_color = "green" if top_idx == 2 else "red"
                st.markdown(f"""
                <h3 style="color:{'green' if top_idx==2 else 'red'};">
                    **{class_names[top_idx]}**
                </h3>
                """, unsafe_allow_html=True)
                
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Confidence chart
                st.subheader("📊 Confidence Scores")
                chart_data = {name: score*100 for name, score in zip(class_names, scores)}
                st.bar_chart(chart_data)
                
                # Debug info
                st.info(f"**Model shape**: {model_shape}")
                st.info(f"**Processed shape**: {predictions.shape}")
    
    else:
        st.info("👈 Upload MRI image to start!")
        st.markdown("### 🧠 Expected Classes:")
        st.markdown("""
        - **Glioma**: Malignant brain tumor
        - **Meningioma**: Meninges tumor  
        - **No Tumor**: Normal
        - **Pituitary**: Pituitary gland tumor
        """)

if __name__ == "__main__":
    main()
