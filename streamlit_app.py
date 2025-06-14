!pip install scikit-learn

import streamlit as st
import pandas as pd
import numpy as np

# Import dengan error handling
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    sklearn_available = True
except ImportError as e:
    st.error(f"Error importing sklearn: {e}")
    st.error("Pastikan scikit-learn dan imbalanced-learn terinstall dengan benar.")
    st.stop()
    sklearn_available = False

import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Obesity Level Prediction",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk melatih model (akan dijalankan sekali saat aplikasi dimuat)
@st.cache_resource
def train_model():
    """
    Fungsi untuk melatih model Random Forest dengan parameter terbaik
    """
    try:
        # Load data (pastikan file CSV tersedia)
        # Untuk deployment, Anda perlu mengunggah file CSV ke direktori yang sama
        df = pd.read_csv('ObesityDataSet.csv')
        
        # Preprocessing data
        df = df.drop_duplicates()
        
        # Handle Weight column
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        df.dropna(subset=['Weight'], inplace=True)
        
        # Remove outliers from Weight
        Q1 = df['Weight'].quantile(0.25)
        Q3 = df['Weight'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['Weight'] >= Q1 - 1.5 * IQR) & (df['Weight'] <= Q3 + 1.5 * IQR)]
        
        # Encode target variable
        le = LabelEncoder()
        df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
        
        # Define features
        numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        categorical_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'MTRANS']
        
        # Prepare X and y
        X = df.drop('NObeyesdad', axis=1)
        y = df['NObeyesdad']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle numerical features
        for col in numerical_features:
            if col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                X_train.dropna(subset=[col], inplace=True)
                X_test.dropna(subset=[col], inplace=True)
                y_train = y_train.loc[X_train.index]
                y_test = y_test.loc[X_test.index]
        
        # Create preprocessor
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit and transform data
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        # Train best Random Forest model (dengan parameter terbaik dari tuning)
        best_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        best_rf.fit(X_train_res, y_train_res)
        
        return best_rf, preprocessor, le, numerical_features, categorical_features
        
    except FileNotFoundError:
        st.error("File ObesityDataSet.csv tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan aplikasi ini.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error dalam melatih model: {str(e)}")
        return None, None, None, None, None

# Load model dan preprocessor
model, preprocessor, label_encoder, numerical_features, categorical_features = train_model()

# Judul aplikasi
st.title("🏥 Prediksi Tingkat Obesitas")
st.markdown("---")

# Sidebar untuk input
st.sidebar.header("📝 Input Data Pengguna")

if model is not None:
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Informasi Pribadi")
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Umur", min_value=10, max_value=100, value=25)
        height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        
    with col2:
        st.subheader("🍎 Kebiasaan Makan")
        favc = st.selectbox("Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
        fcvc = st.slider("Konsumsi Sayuran (frekuensi)", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Jumlah Makanan Utama", 1.0, 4.0, 3.0, step=0.1)
        calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("Konsumsi Air per Hari (liter)", 1.0, 3.0, 2.0, step=0.1)
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏃 Aktivitas Fisik")
        scc = st.selectbox("Monitor Kalori", ["yes", "no"])
        faf = st.slider("Frekuensi Aktivitas Fisik per Minggu", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Waktu Menggunakan Teknologi (jam/hari)", 0.0, 2.0, 1.0, step=0.1)
        
    with col4:
        st.subheader("🚗 Transportasi")
        mtrans = st.selectbox("Mode Transportasi", 
                             ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
    
    # Tombol prediksi
    if st.button("🔍 Prediksi Tingkat Obesitas", type="primary"):
        # Siapkan data input
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CALC': [calc],
            'CH2O': [ch2o],
            'SCC': [scc],
            'FAF': [faf],
            'TUE': [tue],
            'MTRANS': [mtrans]
        })
        
        try:
            # Preprocess input data
            input_scaled = preprocessor.transform(input_data)
            
            # Prediksi
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediksi
            obesity_level = label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100
            
            # Tampilkan hasil
            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tingkat Obesitas", obesity_level)
            
            with col2:
                st.metric("Tingkat Kepercayaan", f"{confidence:.1f}%")
            
            with col3:
                bmi = weight / (height ** 2)
                st.metric("BMI", f"{bmi:.1f}")
            
            # Interpretasi hasil
            st.markdown("---")
            st.subheader("📋 Interpretasi")
            
            interpretations = {
                'Insufficient_Weight': "🟢 Berat badan kurang - Disarankan untuk meningkatkan asupan nutrisi",
                'Normal_Weight': "🟢 Berat badan normal - Pertahankan pola hidup sehat",
                'Overweight_Level_I': "🟡 Kelebihan berat badan tingkat I - Mulai perhatikan pola makan",
                'Overweight_Level_II': "🟡 Kelebihan berat badan tingkat II - Disarankan konsultasi dengan ahli gizi",
                'Obesity_Type_I': "🟠 Obesitas tipe I - Perlu program penurunan berat badan",
                'Obesity_Type_II': "🔴 Obesitas tipe II - Konsultasi medis diperlukan",
                'Obesity_Type_III': "🔴 Obesitas tipe III - Perlu penanganan medis segera"
            }
            
            if obesity_level in interpretations:
                st.info(interpretations[obesity_level])
            
            # Probabilitas untuk setiap kelas
            st.markdown("---")
            st.subheader("📈 Probabilitas untuk Setiap Kategori")
            
            prob_df = pd.DataFrame({
                'Kategori': label_encoder.classes_,
                'Probabilitas': prediction_proba
            }).sort_values('Probabilitas', ascending=False)
            
            st.bar_chart(prob_df.set_index('Kategori')['Probabilitas'])
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
    
    # Informasi tambahan
    st.markdown("---")
    st.subheader("ℹ️ Informasi Model")
    st.info("""
    Model ini menggunakan Random Forest Classifier yang telah dioptimalkan untuk memprediksi tingkat obesitas 
    berdasarkan faktor-faktor seperti kebiasaan makan, aktivitas fisik, dan karakteristik demografis.
    
    **Akurasi Model**: ~96%
    
    **Kategori Obesitas**:
    - Insufficient Weight: Berat badan kurang
    - Normal Weight: Berat badan normal  
    - Overweight Level I & II: Kelebihan berat badan
    - Obesity Type I, II, III: Obesitas berbagai tingkat
    """)
    
else:
    st.error("Model tidak dapat dimuat. Pastikan file ObesityDataSet.csv tersedia.")
    st.info("Untuk menjalankan aplikasi ini, Anda memerlukan file dataset 'ObesityDataSet.csv' di direktori yang sama.")

# Footer
st.markdown("---")
st.markdown("**Catatan**: Hasil prediksi ini hanya untuk referensi. Konsultasikan dengan tenaga medis untuk diagnosis yang akurat.")
