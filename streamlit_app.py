# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings

# Mengabaikan peringatan agar tampilan lebih bersih
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="⚖️",
    layout="wide"
)

# --- Fungsi-Fungsi Pembantu ---

@st.cache_data
def load_data(file):
    """Memuat data dari file CSV yang diunggah."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        return None

def preprocess_data(df_raw):
    """Melakukan semua langkah pra-pemrosesan data."""
    df = df_raw.copy()

    # 1. Hapus duplikat
    df = df.drop_duplicates()

    # 2. Hapus outlier pada kolom 'Weight' menggunakan metode IQR
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df.dropna(subset=['Weight'], inplace=True)
    Q1 = df['Weight'].quantile(0.25)
    Q3 = df['Weight'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Weight'] >= lower_bound) & (df['Weight'] <= upper_bound)]

    # 3. Encoding Target Variable (NObeyesdad)
    le = LabelEncoder()
    df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
    
    # 4. Pemisahan Fitur (X) dan Target (y)
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # 5. Definisikan fitur numerik dan kategorikal
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'MTRANS', 'family_history_with_overweight', 'CAEC', 'SMOKE']
    
    # Pastikan semua kolom ada di DataFrame
    all_features = numerical_features + categorical_features
    X = X[all_features]

    # 6. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 7. Penanganan nilai non-numerik/hilang pada fitur numerik
    for col in numerical_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Imputasi sederhana (mengisi NaN dengan median)
    for col in numerical_features:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
    
    # Imputasi untuk fitur kategorikal (mengisi NaN dengan modus)
    for col in categorical_features:
        mode_val = X_train[col].mode()[0]
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)

    # 8. Transformer untuk preprocessing
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 9. Terapkan preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 10. Terapkan SMOTE untuk menyeimbangkan data training
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

    return X_train_res, y_train_res, X_test_processed, y_test, le

# --- Antarmuka Streamlit ---

st.title("⚖️ Aplikasi Prediksi Tingkat Obesitas")
st.write("""
Aplikasi ini melatih dan mengevaluasi beberapa model machine learning untuk memprediksi tingkat obesitas berdasarkan data gaya hidup dan atribut fisik. 
Unggah file CSV Anda atau gunakan dataset default untuk memulai.
""")

# Sidebar untuk unggah data dan opsi
st.sidebar.header("Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV Anda", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("Menggunakan dataset default. Unggah file untuk menggunakan data Anda.")
    try:
        # Muat dataset default dari direktori lokal
        default_file = 'ObesityDataSet.csv'
        df_raw = load_data(default_file)
    except FileNotFoundError:
        st.error("File 'ObesityDataSet.csv' tidak ditemukan. Harap unggah file atau letakkan file di direktori yang sama dengan `app.py`.")
        st.stop()
else:
    df_raw = load_data(uploaded_file)

if df_raw is not None:
    st.header("EDA (Exploratory Data Analysis) Sederhana")
    
    # Tampilkan beberapa baris data
    if st.checkbox("Tampilkan Sampel Data"):
        st.dataframe(df_raw.head())

    # Tampilkan distribusi kelas target
    if st.checkbox("Tampilkan Distribusi Tingkat Obesitas"):
        st.write("Distribusi kelas target sebelum pra-pemrosesan:")
        fig, ax = plt.subplots()
        sns.countplot(data=df_raw, x="NObeyesdad", order=df_raw["NObeyesdad"].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.header("🚀 Pelatihan dan Evaluasi Model")

    if st.button("Mulai Pelatihan & Evaluasi"):
        with st.spinner('Melakukan pra-pemrosesan data... Ini mungkin memakan waktu beberapa saat.'):
            # Proses data
            X_train_res, y_train_res, X_test_scaled, y_test, label_encoder = preprocess_data(df_raw)
            st.success("Pra-pemrosesan data selesai!")

        # Inisialisasi model
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True)
        }
        
        results = []
        class_names_list = label_encoder.classes_.tolist()

        for name, model in models.items():
            with st.spinner(f"Melatih model {name}..."):
                # Latih model
                model.fit(X_train_res, y_train_res)
                # Prediksi
                y_pred = model.predict(X_test_scaled)

            st.subheader(f"Hasil Evaluasi: {name}")

            # Hitung metrik
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Tampilkan metrik utama dalam kolom
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Akurasi", f"{acc:.2%}")
            col2.metric("Presisi", f"{prec:.2%}")
            col3.metric("Recall", f"{rec:.2%}")
            col4.metric("F1-Score", f"{f1:.2%}")

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })
            
            # Tampilkan classification report dan confusion matrix dalam expander
            with st.expander("Lihat Detail Evaluasi"):
                st.text("Classification Report:")
                report = classification_report(y_test, y_pred, target_names=class_names_list, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.text("Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_names_list, yticklabels=class_names_list, ax=ax)
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot(fig)

        # --- Perbandingan Hasil ---
        st.header("📊 Perbandingan Hasil Antar Model")
        results_df = pd.DataFrame(results).set_index('Model')
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen').format('{:.2%}'))
        
        # Visualisasi perbandingan
        st.subheader("Visualisasi Performa Model")
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        results_df.plot(kind='bar', ax=ax_comp)
        plt.title("Perbandingan Performa Model")
        plt.ylabel("Skor")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        st.pyplot(fig_comp)
