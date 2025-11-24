import streamlit as st
from utils import load_all_models, MODELS # Import shared utilities

# Configure the main application page settings
st.set_page_config(
    page_title="Doctory AI Medical Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Call the model loader once, this will display an error if models/ is missing
if MODELS is None:
    st.error("Application failed to initialize. See console for model loading errors.")
    st.stop() 

# --- HOME PAGE CONTENT ---
st.title("AI Medical Prediction Dashboard")
st.header("Welcome to Doctory AI 🩺")

st.markdown("""
### Use the Sidebar to Select a Specialized Module:
* **AI Chatbot:** Connect to your custom fine-tuned model (Gemma/Gemini) for Q&A.
* **Prediction Modules:** Run local machine learning models for diagnosis based on images or biometric data.
---

### Disclaimer:
**This tool is for informational and educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
""")