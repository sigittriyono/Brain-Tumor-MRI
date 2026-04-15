import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from PIL import Image
from pathlib import Path
import scipy.special  # Untuk softmax

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

def softmax(x):
    """Softmax replacement untuk numpy"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def preprocess_image(image):
    """[1, 240, 240, 3] NHWC"""
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(session, image):
    input_meta = session.get_inputs()[0]
    processed = preprocess_image(image)
    
    predictions = session.run(None, {input_meta.name: processed})[0]
    return predictions

def main():
    st.set_page_config(page_title="Brain Tumor MRI", page_icon="🧠", layout="wide")
    st.title("🧠 Brain Tumor MRI Classifier")
    
    # Model load
    try:
        session = load_model()
        st.success("✅ Model loaded!")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Kontrol")
    show_debug = st.sidebar.checkbox("Show Debug Info")
    
    # Upload
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload MRI", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Image processing
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            elif image_array.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image_array
        else:
            image_bgr = image_array
        
        # Display original
        st.subheader("📸 Original Image")
        st.image(image, use_column_width=True)
        
        # Prediction
        with col2:
            st.subheader("🎯 Prediction Result")
            
            with st.spinner("Analyzing..."):
                predictions = predict(session, image_bgr)
                scores = softmax(predictions[0])  # FIXED!
                
                classes = ['🦠 Glioma', '🎯 Meningioma', '✅ No Tumor', '🦋 Pituitary']
                top_idx = np.argmax(scores)
                confidence = scores[top_idx] * 100
                
                # RESULT CARD
                color = "green" if top_idx == 2 else "red"
                st.markdown(f"""
                <div style='text-align:center;padding:25px;border-radius:15px;
                background:{'linear-gradient(135deg,#d4edda,#f8f9fa)' if top_idx==2 else 'linear-gradient(135deg,#f8d7da,#f8f9fa)'};
                border:3px solid {color};box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                    <h1 style='margin:0;color:{color};font-size:2.5em;'>
                        {classes[top_idx]}
                    </h1>
                    <h2 style='margin:10px 0 0 0;color:{color};opacity:0.9;'>
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            # === BAR CHART PER KELAS ===
            st.subheader("📊 Confidence Scores")
            
            # Custom bar chart dengan matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(classes, scores * 100, 
                         color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
                         alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Highlight top class
            bars[top_idx].set_color('#2ed573' if top_idx==2 else '#ff4757')
            bars[top_idx].set_alpha(1.0)
            
            ax.set_ylabel('Confidence (%)', fontsize=12)
            ax.set_title('Prediksi Brain Tumor - Confidence Scores', fontsize=16, fontweight='bold')
            ax.set_ylim(0, 100)
            
            # Value labels on bars
            for bar, score in zip(bars, scores * 100):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # === DETAIL TABLE ===
            st.subheader("📋 Detail Semua Kelas")
            df_data = {
                'Kelas': [cls.split()[1] for cls in classes],
                'Confidence': [f"{s*100:.1f}%" for s in scores],
                'Probability': [f"{s:.3f}" for s in scores]
            }
            st.dataframe(df_data, use_container_width=True)
        
        # Debug
        if show_debug:
            with st.expander("🔧 Debug Information"):
                input_shape = session.get_inputs()[0].shape
                st.json({
                    "Model input shape": str(input_shape),
                    "Predictions shape": str(predictions.shape),
                    "Raw logits": predictions[0].tolist(),
                    "Softmax scores": scores.tolist(),
                    "Top prediction": f"{classes[top_idx]} ({confidence:.1f}%)"
                })
    
    else:
        st.info("👈 **Upload MRI image** untuk klasifikasi!")
        
        st.markdown("### 🧠 Contoh Hasil:")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://images.unsplash.com/photo-1588735374723-4f0515d4dcbe?w=400", caption="Normal Brain")
        with col2:
            st.image("https://images.unsplash.com/photo-1629429843630-b7c6041f5906?w=400", caption="Brain Tumor")

if __name__ == "__main__":
    main()
