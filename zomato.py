# delivery_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
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
st.markdown("Upload data atau gunakan contoh data. Pilih kolom target (delivery time), pilih fitur/model, lalu train.")

# -------------------------
# helper: generate sample
# -------------------------
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

# -------------------------
# Load data
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV (dataset order) — harus ada kolom target delivery time", type=["csv"])
    use_sample = st.checkbox("Gunakan contoh data (jika tidak upload)", value=True)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil dimuat.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            df = None
    elif use_sample:
        df = generate_sample()
    else:
        df = None

with col2:
    st.markdown("**Model options**")
    default_model = "Random Forest"
    model_choice = st.selectbox(
        "Pilih model",
        options=["Linear Regression", "Random Forest"] + (["XGBoost"] if has_xgb else []) + (["LightGBM"] if has_lgb else []),
        index=1 if default_model in ["Linear Regression", "Random Forest"] else 0
    )
    st.write("XGBoost terinstall:", has_xgb, "LightGBM terinstall:", has_lgb)
    show_shap = st.checkbox("Hitung SHAP (opsional, lebih lama)", value=False)
    max_shap = st.slider("Max sample untuk SHAP (jika aktif)", 50, 1000, 200)

if df is None:
    st.info("Upload dataset CSV atau centang 'Gunakan contoh data' untuk memulai.")
    st.stop()

# -------------------------
# Inspect & choose target
# -------------------------
st.subheader("Preview dataset")
st.dataframe(df.head())

# show columns
st.write("Kolom:", df.columns.tolist())

# detect likely target names
lower_to_orig = {c.lower(): c for c in df.columns}
candidates = ["delivery_time", "time_delivery", "delivery_minutes", "delivery_min", "delivery_time_min"]
detected = None
for cand in candidates:
    if cand in lower_to_orig:
        detected = lower_to_orig[cand]
        break

if detected:
    target_col = st.selectbox("Pilih kolom target (delivery time) — terdeteksi otomatis", options=df.columns, index=list(df.columns).index(detected))
else:
    # fallback: if there is a numeric column named like 'time' or last column numeric, let user choose
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.error("Dataset tidak punya kolom numerik sama sekali — app ini memerlukan kolom target numerik (delivery time).")
        st.stop()
    # prefer 'delivery_time' if exists otherwise let user pick
    default_index = 0 if numeric_cols else 0
    target_col = st.selectbox("Pilih kolom target (kolom numerik yang ingin diprediksi):", options=df.columns, index=default_index)

st.markdown(f"**Target terpilih:** `{target_col}`")

# -------------------------
# Feature selection
# -------------------------
all_features = df.columns.drop(target_col).tolist()
numeric_features = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[all_features].select_dtypes(include=['object','category','bool']).columns.tolist()

st.write("Fitur numerik:", numeric_features)
st.write("Fitur kategorikal:", categorical_features)

selected_features = st.multiselect("Pilih fitur yang dipakai (default: semua)", options=all_features, default=all_features)

if len(selected_features) == 0:
    st.error("Pilih minimal 1 fitur untuk training.")
    st.stop()

# split
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42)

# -------------------------
# Build preprocessing & model
# -------------------------
selected_numeric = [c for c in selected_features if c in numeric_features]
selected_cat = [c for c in selected_features if c in categorical_features]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selected_numeric),
        ("cat", categorical_transformer, selected_cat)
    ],
    remainder="drop"
)

# choose model instance
if model_choice == "Linear Regression":
    model_inst = LinearRegression()
elif model_choice == "Random Forest":
    n_est = st.sidebar.slider("RF: n_estimators", 10, 500, 100, step=10)
    model_inst = RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=random_state)
elif model_choice == "XGBoost" and has_xgb:
    model_inst = XGBRegressor(n_jobs=-1, random_state=random_state, verbosity=0)
elif model_choice == "LightGBM" and has_lgb:
    model_inst = LGBMRegressor(random_state=random_state)
else:
    st.warning("Model tidak tersedia; fallback ke RandomForest.")
    model_inst = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_inst)])

# -------------------------
# Train button
# -------------------------
if st.button("Train model"):
    X = df[selected_features].copy()
    y = df[target_col].copy()

    # quick checks
    if y.isnull().all():
        st.error("Kolom target seluruhnya kosong.")
        st.stop()

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    with st.spinner("Training..."):
        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Gagal training: {e}")
            st.stop()

    # predict
    y_pred = pipeline.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Hasil Evaluasi")
    st.write(f"MAE: **{mae:.3f}**")
    st.write(f"MSE: **{mse:.3f}**")
    st.write(f"RMSE: **{rmse:.3f}**")
    st.write(f"R²: **{r2:.3f}**")

    # scatter plot actual vs pred
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    minv = min(y_test.min(), y_pred.min())
    maxv = max(y_test.max(), y_pred.max())
    ax.plot([minv, maxv], [minv, maxv], color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # feature importance (fast)
    st.subheader("Feature importance (cepat)")
    try:
        model_fitted = pipeline.named_steps["model"]
        # try tree-based feature importances
        if hasattr(model_fitted, "feature_importances_"):
            # get feature names after preprocessing
            try:
                feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                # fallback: create basic names
                feat_names = []
                if selected_numeric:
                    feat_names.extend(selected_numeric)
                if selected_cat:
                    # try derive onehot categories
                    transformer = pipeline.named_steps["preprocessor"].transformers_
                    for name, trans, cols in transformer:
                        if name == "cat":
                            try:
                                ohe = trans.named_steps["onehot"]
                                ohe_names = ohe.get_feature_names_out(cols)
                                feat_names.extend(ohe_names)
                            except Exception:
                                feat_names.extend(cols)
                feat_names = np.array(feat_names)

            importances = model_fitted.feature_importances_
            if len(importances) == len(feat_names):
                fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
                fi_df = fi_df.sort_values("importance", ascending=False).head(20)
                st.table(fi_df.reset_index(drop=True))
                fig2, ax2 = plt.subplots(figsize=(6,4))
                sns.barplot(data=fi_df, x="importance", y="feature", ax=ax2)
                st.pyplot(fig2)
            else:
                st.write("Tidak dapat memetakan nama fitur (panjang beda). Menampilkan top importances (index).")
                fi_sorted = np.sort(importances)[::-1][:20]
                st.write(fi_sorted)
        # linear model: coefficients
        elif hasattr(model_fitted, "coef_"):
            try:
                feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                feat_names = selected_features
            coefs = model_fitted.coef_
            coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
            coef_df = coef_df.sort_values("coef", key=abs, ascending=False).head(20)
            st.table(coef_df.reset_index(drop=True))
        else:
            st.info("Model tidak memiliki cara cepat untuk feature importance.")
    except Exception as e:
        st.error(f"Gagal menampilkan feature importance: {e}")

    # optional: SHAP (slower)
    if show_shap:
        with st.spinner("Menghitung SHAP (disampling untuk mempercepat)..."):
            try:
                import shap
                sample_n = min(max_shap, X_test.shape[0])
                X_train_sample = X_train.sample(min(500, X_train.shape[0]), random_state=42)
                X_test_sample = X_test.sample(sample_n, random_state=42)
                explainer = shap.Explainer(pipeline.named_steps["model"], pipeline.named_steps["preprocessor"].transform(X_train_sample))
                shap_values = explainer(pipeline.named_steps["preprocessor"].transform(X_test_sample))
                st.subheader("SHAP summary (bar)")
                fig_shap = shap.plots.bar(shap_values, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.error(f"SHAP error: {e}")

    # optional: save model
    if st.checkbox("Simpan model ke file (.pkl)?"):
        fname = st.text_input("Nama file (contoh: model_delivery.pkl)", value="model_delivery.pkl")
        try:
            joblib.dump(pipeline, fname)
            st.success(f"Model tersimpan: {fname}")
        except Exception as e:
            st.error(f"Gagal menyimpan model: {e}")
