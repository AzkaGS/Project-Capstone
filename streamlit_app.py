import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and the preprocessor
model = joblib.load('best_random_forest_model.pkl')  # Save your best model using joblib
preprocessor = joblib.load('preprocessor.pkl')  # Save your preprocessor using joblib

# Function to make predictions
def predict(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    input_scaled = preprocessor.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit app layout
st.title("Obesity Level Prediction")
st.write("Enter the following details:")

# Input fields for the features
age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", options=["Male", "Female"])
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
fcvc = st.number_input("Frequency of Vegetables Consumption (0-7)", min_value=0, max_value=7, value=3)
ncp = st.number_input("Number of meals consumed per day", min_value=1, max_value=10, value=3)
ch2o = st.number_input("Water Consumption (liters)", min_value=0.5, max_value=10.0, value=2.0)
faf = st.number_input("Physical Activity Frequency (0-7)", min_value=0, max_value=7, value=3)
tue = st.number_input("Time spent on exercise (hours)", min_value=0, max_value=24, value=1)
calc = st.selectbox("Caloric Intake", options=["Low", "Normal", "High"])
favc = st.selectbox("Frequent Consumption of Fast Food", options=["Yes", "No"])
scc = st.selectbox("Sleep Condition", options=["Good", "Average", "Poor"])
mtrans = st.selectbox("Transportation Mode", options=["Walking", "Biking", "Car", "Public Transport"])

# Button to make prediction
if st.button("Predict"):
    # Prepare input data
    input_data = {
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "FAVC": favc,
        "SCC": scc,
        "MTRANS": mtrans
    }
    
    # Make prediction
    prediction = predict(input_data)
    
    # Display the result
    st.write(f"The predicted obesity level is: {prediction}")

# Run the app
if __name__ == "__main__":
    st.run()
