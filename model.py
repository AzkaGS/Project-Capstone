import pickle
import joblib

# Simpan model terbaik (contoh: Random Forest setelah hyperparameter tuning)
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Simpan scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
