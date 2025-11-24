import streamlit as st
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
import io
import os
import math
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.exceptions import NotFound

# --- 1. MODEL LOADING AND CACHING ---
@st.cache_resource
def load_all_models():
    """Loads all models, scalers, and ONNX sessions from the models/ directory."""
    MODEL_DIR = "models/"
    if not os.path.isdir(MODEL_DIR):
        return None
    
    try:
        # --- Image Models (ONNX) ---
        pneumonia_session = ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        malaria_session = ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))

        # --- Tabular Models (Joblib) ---
        diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_ensemble_model.joblib"))
        diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_scaler.joblib"))
        heart_model = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_model.joblib"))
        heart_scaler = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_scaler.joblib"))
        
        return {
            "pneumonia_session": pneumonia_session,
            "malaria_session": malaria_session,
            "diabetes_model": diabetes_model,
            "diabetes_scaler": diabetes_scaler,
            "heart_model": heart_model,
            "heart_scaler": heart_scaler,
            "pneumonia_input_name": pneumonia_session.get_inputs()[0].name,
            "pneumonia_output_name": pneumonia_session.get_outputs()[0].name,
            "malaria_input_name": malaria_session.get_inputs()[0].name,
            "malaria_output_name": malaria_session.get_outputs()[0].name,
            "pneumonia_classes": ["Normal", "Pneumonia_bacteria", "Pneumonia_virus"]
        }
    except Exception as e:
        st.error(f"Error loading local models. Details: {e}")
        return None

MODELS = load_all_models()

# --------------------------------------------------------------------
# 2. HELPER FUNCTIONS 
# --------------------------------------------------------------------

def process_image_yolo(image_bytes, target_size=(224, 224)):
    """Preprocesses image for the Pneumonia (YOLO-based ONNX) model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def process_image_keras(image_bytes, target_size=(224, 224)):
    """Preprocesses image for the Malaria (Keras-based ONNX) model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0 or height_cm < 50:
        return 0
    return weight_kg / ((height_cm / 100) ** 2)

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

# Feature preparation functions remain here, accessible to all pages:

def prepare_diabetes_features(data):
    # Uses the global scaler loaded in MODELS
    scaler = MODELS['diabetes_scaler']
    age = data.get('Age')
    weight = data.get('Weight')
    height = data.get('Height')
    bp = data.get('BP')
    glucose = data.get('Glucose')
    pregnancies = data.get('Pregnancies', 0)

    bmi = calculate_bmi(height, weight)
    skin_thickness_default = 29.0
    insulin_default = 125.0 
    dpf_default = 0.3725

    feature_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    features = pd.DataFrame([[
        pregnancies, glucose, bp, skin_thickness_default,
        insulin_default, bmi, dpf_default, age
    ]], columns=feature_order)

    return scaler.transform(features)

def prepare_heart_features(data):
    # Uses the global scaler loaded in MODELS
    scaler = MODELS['heart_scaler']
    height = data.get('Height')
    weight = data.get('Weight')
    age = data.get('Age')
    bmi = calculate_bmi(height, weight)
    
    # Mappings (unchanged)
    general_health_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3, 'Very Good': 4}
    checkup_map = {'More than 5 years': 0, 'Never': 1, 'Past 1 year': 2, 'Past 2 years': 3, 'Past 5 years': 4}
    binary_map = {'No': 0, 'Yes': 1} 
    diabetes_map = {'No': 0, 'No Pre Diabetes': 1, 'Only during pregnancy': 2, 'Yes': 3}
    age_category_map = {'Adult': 0, 'Elderly': 1, 'Mid-Aged': 2, 'Senior-Adult': 3, 'Young': 4}
    bmi_group_map = {'Normal weight': 0, 'Obese I': 1, 'Obese II': 2, 'Overweight': 3, 'Underweight': 4}

    # BMI Group Calculation
    bmi_bins = [12.02, 18.3, 26.85, 31.58, 37.8, 100]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese I', 'Obese II']
    try:
        bmi_group_str = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0]
    except ValueError:
        bmi_group_str = 'Normal weight'
        
    # Lifestyle Mappers
    def map_smoking(val): return 1 if val in ['Former', 'Current'] else 0 
    def map_alcohol(val):
        if val == 'Never': return 0
        if val == 'Occasionally': return 4
        if val == 'Weekly': return 8
        if val == 'Daily': return 30
        return 0
    def map_consumption(val):
        if val == '0': return 0
        if val == '1–2': return 12 
        if val == '3–5': return 20 
        if val == '6–7': return 30 
        return 0
    def map_fried(val):
        if val == 'Rarely': return 2
        if val == 'Weekly': return 4
        if val == 'Several times per week': return 8
        return 0

    age_cat_str = get_age_category(age) 

    # Build feature dictionary in the correct final order
    feature_dict = {
        'general_health': general_health_map.get(data.get('General_Health')),
        'checkup': checkup_map.get(data.get('Checkup')),
        'exercise': binary_map.get(data.get('Exercise')), 
        'skin_cancer': binary_map.get(data.get('Skin_Cancer')),
        'other_cancer': binary_map.get(data.get('Other_Cancer')),
        'depression': binary_map.get(data.get('Depression')),
        'diabetes': diabetes_map.get(data.get('Diabetes')),
        'arthritis': binary_map.get(data.get('Arthritis')),
        'age_category': age_category_map.get(age_cat_str),
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'alcohol_consumption': map_alcohol(data.get('Alcohol_Consumption')),
        'fruit_consumption': map_consumption(data.get('Fruit_Consumption')),
        'vegetables_consumption': map_consumption(data.get('Vegetables_Consumption')),
        'potato_consumption': map_fried(data.get('FriedPotato_Consumption')),
        'bmi_group': bmi_group_map.get(bmi_group_str),
        'sex_Female': 1 if data.get('Sex') == 'Female' else 0,
        'sex_Male': 1 if data.get('Sex') == 'Male' else 0,
        'smoking_history_No': 1 if map_smoking(data.get('Smoking_History')) == 0 else 0,
        'smoking_history_Yes': 1 if map_smoking(data.get('Smoking_History')) == 1 else 0,
    }

    final_feature_order = [
        'general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer',
        'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight',
        'bmi', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption',
        'potato_consumption', 'bmi_group', 'sex_Female', 'sex_Male',
        'smoking_history_No', 'smoking_history_Yes'
    ]

    features = pd.DataFrame([feature_dict], columns=final_feature_order)
    return scaler.transform(features)