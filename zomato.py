# delivery_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# optional imports
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

try:
    from lightgbm import LGBMRegressor
    has_lgb = True
except Exception:
    has_lgb = False

st.set_page_config(layout="wide")
st.title("🚀 Delivery Time Prediction — Streamlit")

# ===============================
# Generate sample dataset
# ===============================
@st.cache_data
def generate_sample(n=2000, random_state=42):
    np.random.seed(random_state)
    distance = np.random.exponential(scale=3, size=n)  # km
    num_items = np.random.poisson(1.5, size=n) + 1
    prep_time = np.random.normal(10, 3, size=n).clip(2, 60)
    traffic = np.random.choice(['low', 'medium', 'high'], size=n, p=[0.45, 0.4, 0.15])
    weather = np.random.choice(['clear', 'rain', 'storm'], size=n, p=[0.75, 0.2, 0.05])
    courier_exp = np.random.randint(0, 6, size=n)
    time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=n, p=[0.25, 0.35, 0.3, 0.1])

    base = 10 + distance * 3 + num_items * 1 + prep_time * 0.5
    base += np.where(traffic == 'high', 8, np.where(traffic == 'medium', 4, 0))
    base += np.where(weather == 'rain', 5, np.where(weather == 'storm', 12, 0))
    base -= courier_exp * 0.5
    noise = np.random.normal(0, 4, size=n)
    delivery_time = np.maximum(1, np.round(base + noise)).astype(int)

    df = pd.DataFrame({
        "distance_km": distance.round(2),
        "num_items": num_items,
        "prep_time_min": prep_time.round(1),
        "traffic": traffic,
        "weather": weather,
        "courier_experience_years": courier_exp,
        "time_of_day": time_of_day,
        "delivery_time": delivery_time
    })
    return df

# ===============================
# Load data
# ===============================
uploaded_file = st.file_uploader("Upload CSV (harus ada kolom target delivery time)", type=["csv"])
use_sample = st.checkbox("Gunakan contoh data", value=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil dimuat.")
elif use_sample:
    df = generate_sample()
else:
    st.stop()

st.subheader("Preview dataset")
st.dataframe(df.head())

# ===============================
# Pilih target
# ===============================
target_col = st.selectbox("Pilih kolom target (delivery time)", options=df.columns, index=len(df.columns)-1)

all_features = df.columns.drop(target_col).tolist()
numeric_features = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[all_features].select_dtypes(include=['object','category','bool']).columns.tolist()

st.write("Fitur numerik:", numeric_features)
st.write("Fitur kategorikal:", categorical_features)

selected_features = st.multiselect("Pilih fitur yang dipakai", options=all_features, default=all_features)

if len(selected_features) == 0:
    st.error("Pilih minimal 1 fitur untuk training.")
    st.stop()

# ===============================
# Preprocessing
# ===============================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # ✅ fix disini
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, [c for c in selected_features if c in numeric_features]),
        ("cat", categorical_transformer, [c for c in selected_features if c in categorical_features])
    ]
)

# ===============================
# Model selection
# ===============================
model_choice = st.selectbox("Pilih model", ["Linear Regression", "Random Forest"] + (["XGBoost"] if has_xgb else []) + (["LightGBM"] if has_lgb else []))

if model_choice == "Linear Regression":
    model_inst = LinearRegression()
elif model_choice == "Random Forest":
    model_inst = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "XGBoost" and has_xgb:
    model_inst = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
elif model_choice == "LightGBM" and has_lgb:
    model_inst = LGBMRegressor(random_state=42)
else:
    model_inst = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_inst)])

# ===============================
# Train & evaluate
# ===============================
if st.button("Train model"):
    X = df[selected_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Hasil Evaluasi")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.2f}")

    # Scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    st.pyplot(fig)
