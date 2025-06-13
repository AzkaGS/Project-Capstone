import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Obesity Level Prediction", page_icon="🏋️‍♂️", layout="centered")

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('best_random_forest_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

st.title("Obesity Level Prediction")
st.write("Fill in the following details to predict obesity level:")

# User inputs
def user_input_features():
    Age = st.number_input("Age", min_value=0, max_value=120, value=25)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    Weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    FCVC = st.number_input("Frequency of Vegetable Consumption (0-7)", min_value=0, max_value=7, value=3)
    NCP = st.number_input("Number of Meals Per Day", min_value=1, max_value=10, value=3)
    CH2O = st.number_input("Water Consumption (Liters/day)", min_value=0.1, max_value=10.0, value=2.0, format="%.2f")
    FAF = st.number_input("Physical Activity Frequency (0-7)", min_value=0, max_value=7, value=3)
    TUE = st.number_input("Time Spent on Exercise (hours/day)", min_value=0, max_value=24, value=1)
    CALC = st.selectbox("Caloric Intake", ["Low", "Normal", "High"])
    FAVC = st.selectbox("Frequent Consumption of Fast Food", ["Yes", "No"])
    SCC = st.selectbox("Sleep Condition", ["Good", "Average", "Poor"])
    MTRANS = st.selectbox("Transportation Mode", ["Walking", "Biking", "Car", "Public Transport"])
    
    data = {
        "Age": Age,
        "Gender": Gender,
        "Height": Height,
        "Weight": Weight,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "CALC": CALC,
        "FAVC": FAVC,
        "SCC": SCC,
        "MTRANS": MTRANS,
    }
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict"):
    # Preprocess and predict
    try:
        input_processed = preprocessor.transform(input_df)
        prediction_encoded = model.predict(input_processed)[0]
        
        # Map encoded labels to original labels
        label_map = {
            0: 'Insufficient Weight',
            1: 'Normal Weight',
            2: 'Overweight Level I',
            3: 'Overweight Level II',
            4: 'Obese Level I',
            5: 'Obese Level II',
            6: 'Obese Level III',
        }
        prediction_label = label_map.get(prediction_encoded, "Unknown")
        
        st.success(f"Predicted Obesity Level: {prediction_label}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
