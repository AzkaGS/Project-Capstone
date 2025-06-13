import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan preprocessor
model = joblib.load('obesity_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")

st.title("🎯 Prediksi Tingkat Obesitas")
st.markdown("Masukkan informasi berikut untuk memprediksi tingkat obesitas Anda.")

# Input dari pengguna
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 10, 100, 25)
height = st.number_input("Height (meter)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
favc = st.selectbox("Apakah sering mengonsumsi makanan tinggi kalori?", ['yes', 'no'])
fcvc = st.slider("Frekuensi konsumsi sayur (0-3)", 0.0, 3.0, 2.0, step=0.1)
ncp = st.slider("Jumlah makan per hari", 1.0, 5.0, 3.0, step=0.1)
ca = st.selectbox("Konsumsi alkohol", ['no', 'Sometimes', 'Frequently'])
ch2o = st.slider("Jumlah konsumsi air (0-3 liter)", 0.0, 3.0, 2.0, step=0.1)
faf = st.slider("Frekuensi aktivitas fisik (0-3 jam/minggu)", 0.0, 3.0, 1.0, step=0.1)
tue = st.slider("Waktu depan komputer/gadget per hari (0-2 jam)", 0.0, 2.0, 1.0, step=0.1)
scc = st.selectbox("Ada penyakit kronis?", ['yes', 'no'])
mtrans = st.selectbox("Transportasi utama", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Prediksi
if st.button("🔍 Prediksi Obesitas"):
    # Data user dalam DataFrame
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': 'no',  # bisa diubah jika tersedia
        'SMOKE': 'no',  # default
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': ca,
        'MTRANS': mtrans
    }])

    # Kolom yang dipakai oleh model
    input_data = input_data[[
        'Gender', 'Age', 'Height', 'Weight', 'FAVC', 'FCVC',
        'NCP', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ]]

    # Preprocessing
    input_scaled = preprocessor.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    prediction_label = le.inverse_transform([prediction])[0]

    st.success(f"✅ Prediksi Tingkat Obesitas Anda: **{prediction_label}**")
