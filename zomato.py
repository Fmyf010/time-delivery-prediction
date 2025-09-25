import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================
# STREAMLIT APP
# ==============================

st.title("🍽️ Zomato Restaurant Rating Prediction")
st.markdown("Analisis dataset Zomato & prediksi rating restoran dengan berbagai model ML.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload dataset Zomato (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Tampilkan kolom yang tersedia
    st.write("**Kolom tersedia:**", df.columns.tolist())

    # Pilih target kolom
    if "rating" in df.columns:
        target_col = "rating"
    else:
        target_col = st.selectbox("Pilih kolom target (nilai yang ingin diprediksi):", df.columns)

    # Distribusi target
    st.subheader(f"Distribusi Target: {target_col}")
    fig, ax = plt.subplots()
    sns.histplot(df[target_col], kde=True, ax=ax, bins=20, color="skyblue")
    st.pyplot(fig)

    # Preprocessing: ambil fitur numerik
    X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
    y = df[target_col]

    if X.shape[1] == 0:
        st.error("Dataset tidak punya fitur numerik untuk dipakai training.")
    else:
        # Split data
        test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Pilih model
        st.sidebar.subheader("Pilih Model")
        model_name = st.sidebar.selectbox(
            "Model",
            ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Random Forest", "XGBoost", "LightGBM"]
        )

        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge()
        elif model_name == "Lasso":
            model = Lasso()
        elif model_name == "ElasticNet":
            model = ElasticNet()
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
        elif model_name == "XGBoost":
            model = XGBRegressor()
        elif model_name == "LightGBM":
            model = LGBMRegressor()

        # Training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("Evaluasi Model")
        st.write("**MAE:**", round(mean_absolute_error(y_test, y_pred), 4))
        st.write("**MSE:**", round(mean_squared_error(y_test, y_pred), 4))
        st.write("**R²:**", round(r2_score(y_test, y_pred), 4))

        # SHAP (Feature Importance) -> optional & sampling
        if st.checkbox("Hitung Feature Importance (SHAP)?"):
            with st.spinner("Menghitung SHAP values, mohon tunggu..."):
                sample_size = min(200, X_train.shape[0])  # max 200 rows
                X_train_sample = X_train.sample(sample_size, random_state=42)
                X_test_sample = X_test.sample(min(200, X_test.shape[0]), random_state=42)

                try:
                    explainer = shap.Explainer(model, X_train_sample)
                    shap_values = explainer(X_test_sample)

                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal menghitung SHAP: {e}")
else:
    st.info("Silakan upload dataset Zomato dalam format CSV untuk memulai analisis.")
