from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load(r'C:\Users\saikeerthana\OneDrive\Desktop\projects\diabetesApp\models\logreg_model.joblib')

# Define thresholds directly in the code
THRESHOLDS = {
    'Glucose': {'min': 70, 'max': 200},
    'BloodPressure': {'min': 60, 'max': 140},
    'BMI': {'min': 18.5, 'max': 40},
    'Age': {'min': 0, 'max': 120},
    'Insulin': {'min': 0, 'max': 850},
    'SkinThickness': {'min': 0, 'max': 100},
    'DiabetesPedigreeFunction': {'min': 0, 'max': 2.5},
    'Pregnancies': {'min': 0, 'max': 20}
}

class DiabetesData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    @field_validator('*')
    @classmethod
    def validate_fields(cls, value: float, info):
        field_name = info.field_name
        if field_name in THRESHOLDS:
            limits = THRESHOLDS[field_name]
            if value < limits['min'] or value > limits['max']:
                raise ValueError(
                    f"{field_name} should be between {limits['min']} and {limits['max']}"
                )
        return value

app = FastAPI()

@app.post("/predict")
def predict(data: DiabetesData):
    input_data = {
        'Pregnancies': [data.Pregnancies],
        'Glucose': [data.Glucose],
        'BloodPressure': [data.BloodPressure],
        'SkinThickness': [data.SkinThickness],
        'Insulin': [data.Insulin],
        'BMI': [data.BMI],
        'DiabetesPedigreeFunction': [data.DiabetesPedigreeFunction],
        'Age': [data.Age]
    }
    input_df = pd.DataFrame(input_data)

    # Make prediction
    prediction_proba = model.predict_proba(input_df)[0]
    prediction = 1 if prediction_proba[1] > 0.6 else 0  # Using a higher threshold for positive prediction
    
    result = "Diabetes" if prediction == 1 else "Not Diabetes"
    confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
    
    risk_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
    
    return {
        "prediction": result,
        "confidence": f"{confidence:.2%}",
        "risk_level": risk_level,
        "message": "Please note that this is just a prediction and should not be used as a medical diagnosis. Consult a healthcare professional for proper medical advice."
    }