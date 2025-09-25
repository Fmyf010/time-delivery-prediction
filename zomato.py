import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Judul Aplikasi ===
st.title("🍽️ Zomato Restaurant Rating Prediction")

# === Upload Dataset ===
uploaded_file = st.file_uploader("Upload dataset Zomato (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.write(df.head())

    # === EDA sederhana ===
    st.subheader("Distribusi Rating")
    fig, ax = plt.subplots()
    sns.histplot(df["rating"], kde=True, ax=ax)
    st.pyplot(fig)

    # === Preprocessing + Training Model ===
    X = df.drop("rating", axis=1, errors="ignore").select_dtypes(include=[np.number])
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # === Evaluasi ===
    st.subheader("Evaluasi Model (Linear Regression)")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R²:", r2_score(y_test, y_pred))
