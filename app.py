from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Diabetes Prediction", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

# Same column order used when the scaler/model were trained in the notebook
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

st.title("Diabetes Prediction using Logistic Regression")
st.write("Enter the patient details below and click **Predict**.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", 1.0, 300.0, 120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 1.0, 200.0, 70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", 1.0, 100.0, 23.0)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 1.0, 900.0, 30.0)
    bmi = st.number_input("BMI", 10.0, 70.0, 32.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.47, step=0.01)
    age = st.number_input("Age (years)", 1, 120, 33, step=1)

if st.button("Predict", type="primary"):
    row = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=FEATURES,
    )
    scaled = scaler.transform(row)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])

    st.subheader("Result")
    if pred == 1:
        st.error(f"Likely DIABETIC (probability {prob:.1%})")
    else:
        st.success(f"Likely NOT diabetic (probability of diabetes {prob:.1%})")
    st.progress(prob)
    st.caption("Educational project only - not a medical diagnosis.")
