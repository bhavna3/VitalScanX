import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/predict"

st.title("Diabetes Prediction App")
st.write("Enter the details below to make a prediction.")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=0, step=1)

if st.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    response = requests.post(API_URL, data=json.dumps(input_data), headers={"Content-Type": "application/json"})
   
    if response.status_code == 200:
        prediction = response.json().get("prediction", "No prediction")
        st.success(f"Prediction: {prediction}")
    else:
        st.error("Error in making prediction. Please check your input data and try again.")

        