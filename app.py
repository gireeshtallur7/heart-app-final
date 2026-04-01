import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction")
st.title("❤️ Heart Disease Prediction App")

# Load model
try:
    model = joblib.load("logistic_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_cols = joblib.load("heart_Columns.pkl")
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting BP", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting BS >120", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
old_peak = st.slider("Old Peak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction
if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": old_peak
    }

    raw_input["Sex_Male"] = 1 if sex == "Male" else 0
    raw_input["Sex_Female"] = 1 if sex == "Female" else 0

    raw_input[f"ChestPainType_{chest_pain_type}"] = 1
    raw_input[f"RestingECG_{resting_ecg}"] = 1
    raw_input[f"ExerciseAngina_{exercise_angina}"] = 1
    raw_input[f"ST_Slope_{st_slope}"] = 1

    df = pd.DataFrame([raw_input])

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")