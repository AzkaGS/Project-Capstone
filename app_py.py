import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Obesity Classification App",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Obesity Classification System</h1>
    <p>Prediksi Tingkat Obesitas Berdasarkan Gaya Hidup dan Kebiasaan Makan</p>
    <p><em>Capstone Project - Bengkel Koding Data Science</em></p>
</div>
""", unsafe_allow_html=True)

# Load models (untuk demo, kita buat mock functions)
@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    try:
        # Dalam implementasi nyata, gunakan:
        # model = joblib.load('best_obesity_model.pkl')
        # scaler = joblib.load('scaler.pkl')
        # target_encoder = joblib.load('target_encoder.pkl')
        
        # Untuk demo, return None (akan dibuat mock prediction)
        return None, None, None
    except:
        return None, None, None

def predict_obesity(features, model, scaler, target_encoder):
    """Make prediction"""
    if model is None:
        # Mock prediction untuk demo
        predictions = ['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I', 
                      'Obesity_Type_II', 'Obesity_Type_III', 'Insufficient_Weight']
        probabilities = np.random.dirichlet(np.ones(7), size=1)[0]
        return np.random.choice(predictions), probabilities
    
    # Implementasi nyata
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    prediction_label = target_encoder.inverse_transform([prediction])[0]
    return prediction_label, probabilities

# Load models
model, scaler, target_encoder = load_models()

# Sidebar untuk input
st.sidebar.header("📝 Input Data Pasien")

# Informasi demografis
st.sidebar.subheader("Informasi Demografis")
gender = st.sidebar.selectbox("Jenis Kelamin", ["Female", "Male"])
age = st.sidebar.slider("Usia", 14, 61, 25)
height = st.sidebar.slider("Tinggi Badan (m)", 1.45, 1.98, 1.70, 0.01)
weight = st.sidebar.slider("Berat Badan (kg)", 39, 173, 70)

# Riwayat keluarga
st.sidebar.subheader("Riwayat Keluarga")
family_history = st.sidebar.selectbox(
    "Riwayat Keluarga dengan Kelebihan Berat Badan", 
    ["yes", "no"]
)

# Kebiasaan makan
st.sidebar.subheader("Kebiasaan Makan")
favc = st.sidebar.selectbox(
    "Sering Mengonsumsi Makanan Tinggi Kalori", 
    ["yes", "no"]
)
fcvc = st.sidebar.slider("Frekuensi Makan Sayuran (per hari)", 1, 3, 2)
ncp = st.sidebar.slider("Jumlah Makanan Utama (per hari)", 1.0, 4.0, 3.0, 0.1)
caec = st.sidebar.selectbox(
    "Makan Camilan di Antara Waktu Makan", 
    ["no", "Sometimes", "Frequently", "Always"]
)

# Gaya hidup
st.sidebar.subheader("Gaya Hidup")
smoke = st.sidebar.selectbox("Merokok", ["yes", "no"])
ch2o = st.sidebar.slider("Konsumsi Air (liter/hari)", 1.0, 3.0, 2.0, 0.1)
scc = st.sidebar.selectbox("Memantau Asupan Kalori", ["yes", "no"])
faf = st.sidebar.slider("Frekuensi Aktivitas Fisik (per minggu)", 0.0, 3.0, 1.0, 0.1)
tue = st.sidebar.slider("Penggunaan Teknologi (jam/hari)", 0, 2, 1)
calc = st.sidebar.selectbox(
    "Konsumsi Alkohol", 
    ["no", "Sometimes", "Frequently", "Always"]
)
mtrans = st.sidebar.selectbox(
    "Jenis Transportasi", 
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)

# Tombol prediksi
predict_button = st.sidebar.button("🔮 Prediksi Tingkat Obesitas", type="primary")

# Main content
col1, col2 = st.columns([2,
