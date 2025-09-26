# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================
# Load Data
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("Zomato.csv")
    return df

# =============================
# Train Model
# =============================
def train_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=200, random_state=42)
    else:
        raise ValueError("Model tidak dikenali!")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluasi
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return model, preds, mae, rmse, r2

# =============================
# Main App
# =============================
st.set_page_config(page_title="Zomato Delivery Prediction", layout="wide")

st.title("🚴 Zomato Delivery Time Prediction App")
st.markdown("Aplikasi interaktif untuk memprediksi waktu antar makanan berdasarkan dataset Zomato.")

# Sidebar menu
menu = st.sidebar.radio("Navigasi", ["EDA", "Training Model", "Prediksi"])

df = load_data()

# =============================
# EDA
# =============================
if menu == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.write("### Preview Dataset")
    st.dataframe(df.head())

    st.write("### Ringkasan Statistik")
    st.write(df.describe())

    st.write("### Distribusi Target (Time_taken(min))")
    fig, ax = plt.subplots()
    sns.histplot(df["Time_taken(min)"], kde=True, bins=20, ax=ax)
    st.pyplot(fig)

    st.write("### Korelasi Fitur dengan Target")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

# =============================
# Training Model
# =============================
elif menu == "Training Model":
    st.header("🤖 Training Model")

    target = "Time_taken(min)"
    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_name = st.selectbox("Pilih Model", ["Random Forest", "Linear Regression", "XGBoost"])

    if st.button("Train"):
        model, preds, mae, rmse, r2 = train_model(model_name, X_train, y_train, X_test, y_test)

        st.success(f"Model {model_name} selesai dilatih!")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R²:** {r2:.3f}")

        # Plot Prediksi vs Aktual
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Aktual")
        ax.set_ylabel("Prediksi")
        ax.set_title("Aktual vs Prediksi")
        st.pyplot(fig)

        # Simpan model di session state
        st.session_state["model"] = model
        st.session_state["features"] = X.columns.tolist()

# =============================
# Prediksi
# =============================
elif menu == "Prediksi":
    st.header("🔮 Prediksi Waktu Antar")

    if "model" not in st.session_state:
        st.warning("Latih model terlebih dahulu di menu **Training Model**.")
    else:
        model = st.session_state["model"]
        feature_names = st.session_state["features"]

        # Buat form input
        st.write("Masukkan data order:")
        user_input = {}
        for col in feature_names:
            if "_Yes" in col or "_1" in col:  # kategori hasil dummy
                user_input[col] = st.selectbox(col, [0, 1])
            else:
                user_input[col] = st.number_input(col, value=0.0)

        input_df = pd.DataFrame([user_input])

        if st.button("Prediksi"):
            pred = model.predict(input_df)[0]
            st.success(f"⏱️ Estimasi waktu antar: **{pred:.2f} menit**")
