import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pip install -r requirements.txt

# ==============================
# STREAMLIT APP
# ==============================

st.title("🍽️ Zomato Restaurant Rating Prediction")
st.markdown("Analisis data Zomato & prediksi rating restoran menggunakan Machine Learning.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload dataset Zomato (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Basic Info
    st.write("**Shape dataset:**", df.shape)
    st.write("**Missing values:**")
    st.write(df.isnull().sum())

    # Distribusi rating
    if "rating" in df.columns:
        st.subheader("Distribusi Rating")
        fig, ax = plt.subplots()
        sns.histplot(df["rating"], kde=True, ax=ax, bins=20, color="skyblue")
        st.pyplot(fig)

        # Preprocessing sederhana (numerical only)
        X = df.drop("rating", axis=1).select_dtypes(include=[np.number])
        y = df["rating"]

        # Split data
        test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

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

        # SHAP (Feature Importance)
        st.subheader("Feature Importance (SHAP)")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)

    else:
        st.error("Kolom `rating` tidak ditemukan di dataset!")
else:
    st.info("Silakan upload dataset Zomato dalam format CSV untuk memulai analisis.")
