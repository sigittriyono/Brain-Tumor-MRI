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
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    if not Path(MODEL_PATH).exists():
        with st.spinner("📥 Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    model_path = download_model()
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image):
    """EXACT model format: [1, 240, 240, 3] NHWC"""
    # Resize to 240x240
    image = cv2.resize(image, (240, 240))
    # Normalize 0-1
    image = image.astype(np.float32) / 255.0
    # Add batch dimension -> [1, 240, 240, 3]
    image = np.expand_dims(image, axis=0)
    return image

def predict(session, image):
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    st.info(f"🎯 Model expects: {input_shape}")
    
    # Preprocess EXACTLY as model expects
    processed = preprocess_image(image)
    st.info(f"📦 Input shape: {processed.shape}")
    
    predictions = session.run(None, {input_name: processed})[0]
    return predictions

def main():
    st.set_page_config(page_title="Brain Tumor MRI", layout="wide")
    st.title("🧠 Brain Tumor Classifier")
    
    # Load model
    try:
        session = load_model()
        input_shape = session.get_inputs()[0].shape
        st.success(f"✅ Loaded! Model input: `[batch, {input_shape[1]}, {input_shape[2]}, {input_shape[3]}]`")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    # === UPLOAD ===
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload MRI", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to BGR for cv2
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Prediction")
            
            with st.spinner("Analyzing..."):
                predictions = predict(session, image_bgr)
                scores = np.softmax(predictions[0])
                
                # Classes
                classes = ['🦠 Glioma', '🎯 Meningioma', '✅ No Tumor', '🦋 Pituitary']
                top_idx = np.argmax(scores)
                confidence = scores[top_idx]
                
                # RESULT
                st.markdown(f"""
                <div style='text-align:center; padding:20px; 
                border-radius:10px; 
                background-color:{'lightgreen' if top_idx==2 else '#ffcccc'}; 
                border:3px solid {'green' if top_idx==2 else 'red'};'>
                    <h2 style='margin:0; color:{'green' if top_idx==2 else 'red'}'>
                        {classes[top_idx]}
                    </h2>
                    <h3 style='margin:10px 0 0 0; opacity:0.8'>
                        Confidence: {confidence*100:.1f}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart
                st.subheader("📊 All Probabilities")
                chart_data = dict(zip([c.split()[1] for c in classes], scores*100))
                st.bar_chart(chart_data)
                
                # Debug
                with st.expander("🔧 Debug Info"):
                    st.json({
                        "Model input shape": str(session.get_inputs()[0].shape),
                        "Input data shape": str(predictions.shape),
                        "Raw scores": {classes[i]: f"{s*100:.2f}%" for i,s in enumerate(scores)}
                    })
    
    else:
        st.info("👈 **Upload MRI scan** to classify!")
        st.markdown("""
        ### 🎯 **4 Classes:**
        """)
        
