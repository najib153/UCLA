import os
import streamlit as st
import joblib

# Try multiple possible locations for the model files
MODEL_PATHS = [
    "model/mlp_model.pkl",  # Local development
    "ucla/model/mlp_model.pkl",  # If in subfolder
    "/mount/src/ucla/model/mlp_model.pkl"  # Streamlit Cloud path
]

@st.cache_resource
def load_model():
    """Try loading model from multiple possible locations"""
    for path in MODEL_PATHS:
        try:
            model = joblib.load(path)
            st.success(f"Successfully loaded model from: {path}")
            return model
        except FileNotFoundError:
            continue
    st.error(f"Model not found in any of these locations: {MODEL_PATHS}")
    st.error("Please ensure:")
    st.error("1. The 'model' folder exists")
    st.error("2. It contains mlp_model.pkl")
    st.error("3. Files are committed to your repository")
    st.stop()

# Load models
try:
    mlp_model = load_model()
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
