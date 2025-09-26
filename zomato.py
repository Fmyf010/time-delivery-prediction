# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ============ CONFIG ============ #
st.set_page_config(page_title="Zomato Delivery Time Predictor", 
                   page_icon="⏱️", 
                   layout="centered")

st.title("⏱️ Prediksi Waktu Pengantaran Zomato")
st.write("Aplikasi ini memprediksi estimasi waktu pengantaran makanan berdasarkan data historis Zomato.")

# ============ LOAD & TRAIN MODEL ============ #
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("Zomato.csv")   # pastikan file ada di folder yang sama

    # Tentukan target
    target = "Delivery_Time"  # ganti sesuai nama kolom target kamu
    X = df.drop(columns=[target])
    y = df[target]

    # Label Encoding untuk kolom kategorikal
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Model Random Forest
    model = RandomForestRegressor(
        n_estimators=200, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, encoders, X.columns.tolist()

model, encoders, features = load_and_train()

# ============ INPUT FORM ============ #
st.header("🔽 Masukkan Detail Pesanan")

user_input = {}
for col in features:
    # Jika kolom kategori → selectbox
    if col in encoders:
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0, value=1)

# ============ PREDICTION ============ #
if st.button("Prediksi Waktu Pengantaran 🚀"):
    with st.spinner("Menghitung estimasi..."):
        # Buat dataframe input
        input_df = pd.DataFrame([user_input])

        # Apply encoder untuk kategori
        for col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

        # Prediksi
        prediction = model.predict(input_df)[0]
    
    st.success(f"Estimasi waktu pengantaran: **{round(prediction, 2)} menit** 🍽️")
    st.balloons()

# ============ FOOTER ============ #
st.markdown("---")
st.caption("Dibuat untuk Tugas Akhir • Zomato Delivery Time Prediction")
