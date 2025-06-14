import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan scaler
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Mapping untuk hasil prediksi
obesity_levels = {
    0: "Berat Badan Kurang (Insufficient Weight)",
    1: "Berat Badan Normal (Normal Weight)", 
    2: "Kelebihan Berat Badan Tingkat I (Overweight Level I)",
    3: "Kelebihan Berat Badan Tingkat II (Overweight Level II)",
    4: "Obesitas Tipe I (Obesity Type I)",
    5: "Obesitas Tipe II (Obesity Type II)",
    6: "Obesitas Tipe III (Obesity Type III)"
}

# Sidebar untuk input
st.sidebar.header("📊 Input Data Prediksi")
st.sidebar.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas:")

# Input features
col1, col2 = st.columns(2)

with col1:
    gender = st.sidebar.selectbox("Jenis Kelamin", ["Female", "Male"])
    age = st.sidebar.slider("Usia", 14, 61, 25)
    height = st.sidebar.slider("Tinggi Badan (m)", 1.45, 1.98, 1.70, 0.01)
    weight = st.sidebar.slider("Berat Badan (kg)", 39.0, 173.0, 70.0, 0.1)
    
with col2:
    family_history = st.sidebar.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
    favc = st.sidebar.selectbox("Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
    fcvc = st.sidebar.slider("Konsumsi Sayuran (per hari)", 1, 3, 2)
    ncp = st.sidebar.slider("Jumlah Makan Utama", 1.0, 4.0, 3.0, 0.1)

# Input lainnya
caec = st.sidebar.selectbox("Makan Camilan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.sidebar.selectbox("Merokok", ["yes", "no"])
ch2o = st.sidebar.slider("Konsumsi Air (liter/hari)", 1.0, 3.0, 2.0, 0.1)
scc = st.sidebar.selectbox("Monitor Kalori", ["yes", "no"])
faf = st.sidebar.slider("Frekuensi Aktivitas Fisik", 0.0, 3.0, 1.0, 0.1)
tue = st.sidebar.slider("Penggunaan Teknologi (jam/hari)", 0, 2, 1)
calc = st.sidebar.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.sidebar.selectbox("Transportasi", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# Main content
st.title("🏥 Sistem Prediksi Tingkat Obesitas")
st.markdown("---")

# Tampilkan informasi dataset
with st.expander("ℹ️ Informasi Dataset"):
    st.write("""
    Dataset ini memuat informasi dari 2111 individu dari tiga negara (Meksiko, Peru, dan Kolombia) 
    dengan 17 atribut yang digunakan untuk memprediksi tingkat obesitas berdasarkan kebiasaan makan 
    dan kondisi fisik.
    
    **Kategori Obesitas:**
    - Berat Badan Kurang (Insufficient Weight)
    - Berat Badan Normal (Normal Weight)
    - Kelebihan Berat Badan Tingkat I & II
    - Obesitas Tipe I, II, dan III
    """)

# Tombol prediksi
if st.sidebar.button("🔍 Prediksi Tingkat Obesitas", type="primary"):
    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    family_history_encoded = 1 if family_history == "yes" else 0
    favc_encoded = 1 if favc == "yes" else 0
    smoke_encoded = 1 if smoke == "yes" else 0
    scc_encoded = 1 if scc == "yes" else 0
    
    # Encode CAEC
    caec_mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    caec_encoded = caec_mapping[caec]
    
    # Encode CALC
    calc_mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_encoded = calc_mapping[calc]
    
    # Encode MTRANS
    mtrans_mapping = {"Automobile": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Walking": 4}
    mtrans_encoded = mtrans_mapping[mtrans]
    
    # Buat array input
    input_data = np.array([[
        gender_encoded, age, height, weight, family_history_encoded,
        favc_encoded, fcvc, ncp, caec_encoded, smoke_encoded,
        ch2o, scc_encoded, faf, tue, calc_encoded, mtrans_encoded
    ]])
    
    # Normalisasi data
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Tampilkan hasil
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 Hasil Prediksi")
        
        # BMI calculation
        bmi = weight / (height ** 2)
        
        # Display BMI
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "blue"
        elif bmi < 25:
            bmi_status = "Normal"
            bmi_color = "green"
        elif bmi < 30:
            bmi_status = "Overweight"
            bmi_color = "orange"
        else:
            bmi_status = "Obese"
            bmi_color = "red"
        
        st.metric("BMI", f"{bmi:.1f}", f"{bmi_status}")
        st.markdown(f"<p style='color: {bmi_color};'>Status BMI: {bmi_status}</p>", unsafe_allow_html=True)
        
        # Hasil prediksi utama
        result = obesity_levels[prediction]
        confidence = max(prediction_proba) * 100
        
        st.success(f"**Prediksi:** {result}")
        st.info(f"**Confidence:** {confidence:.1f}%")
        
        # Bar chart probabilitas
        st.markdown("### 📊 Probabilitas Setiap Kategori")
        prob_df = pd.DataFrame({
            'Kategori': [obesity_levels[i] for i in range(len(prediction_proba))],
            'Probabilitas': prediction_proba
        })
        st.bar_chart(prob_df.set_index('Kategori')['Probabilitas'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Capstone Project - Bengkel Koding Data Science</p>
    <p>Universitas Dian Nuswantoro | Program Studi Teknik Informatika</p>
</div>
""", unsafe_allow_html=True)
