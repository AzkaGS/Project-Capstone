import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
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
        best_r
