import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_model.pkl")

st.title("🩺 Heart Disease Prediction App")

# User inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1])  
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.5, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.slider("Number of Vessels (0-3)", 0, 3, 0)
thal = st.selectbox("Thal (0 = normal, 1 = fixed defect, 2 = reversible)", [0, 1, 2])

# Prepare input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ No Heart Disease detected")
