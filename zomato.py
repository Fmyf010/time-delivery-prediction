import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Fungsi load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv")  # ganti dengan dataset Anda
    return df

# -----------------------------
# Aplikasi Streamlit
# -----------------------------
def main():
    st.title("🚚 Prediksi Delivery Time - Zomato")

    # Load data
    df = load_data()
    st.subheader("Data Awal")
    st.write(df.head())

    # Pastikan target ada
    if "delivery_time" not in df.columns:
        st.error("Kolom 'delivery_time' tidak ada di dataset!")
        return

    # Split features dan target
    X = df.drop("delivery_time", axis=1)
    y = df["delivery_time"]

    # Pisahkan kolom numerik & kategorikal
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Pipeline preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Pipeline model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    st.subheader("Evaluasi Model")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Prediksi manual
    st.subheader("Coba Prediksi")
    user_input = {}
    for col in num_cols:
        user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
    for col in cat_cols:
        user_input[col] = st.selectbox(f"{col}", df[col].unique())

    if st.button("Prediksi Delivery Time"):
        user_df = pd.DataFrame([user_input])
        pred = model.predict(user_df)[0]
        st.success(f"Estimasi waktu pengiriman: {pred:.2f} menit")

if __name__ == "__main__":
    main()
