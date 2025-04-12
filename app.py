import os
import streamlit as st
import joblib

# Debug: List files in directory
st.write("Current directory contents:", os.listdir())
if os.path.exists("model"):
    st.write("Model folder contents:", os.listdir("model"))

@st.cache_resource
def load_model():
    """Try loading model with multiple fallback paths"""
    possible_paths = [
        "model/mlp_model.pkl",  # Standard location
        os.path.join(os.path.dirname(__file__), "model/mlp_model.pkl"),  # Absolute path
        "/mount/src/ucla/model/mlp_model.pkl"  # Streamlit Cloud path
    ]
    
    for path in possible_paths:
        try:
            return joblib.load(path)
        except:
            continue
    
    st.error("""
    Model file not found. Please ensure:
    1. The 'model' folder exists
    2. It contains mlp_model.pkl
    3. Files are committed to your repository
    """)
    st.stop()

try:
    mlp_model = load_model()
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()
