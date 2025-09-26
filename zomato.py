import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from haversine import haversine, Unit
import warnings

# --- Konfigurasi Dasar Aplikasi ---
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Prediksi Waktu Pengiriman Zomato",
    page_icon="🛵",
    layout="wide"
)

# --- Fungsi untuk Memuat, Membersihkan, dan Melatih Model ---
@st.cache_resource
def load_and_train_model(data_path):
    """
    Fungsi ini memuat data, membersihkannya, melakukan feature engineering,
    dan melatih model machine learning.
    """
    # 1. Memuat Data
    df = pd.read_csv(data_path)

    # 2. Pembersihan Data dan Feature Engineering (BAGIAN YANG DIPERBAIKI)
    # Membuat nama kolom menjadi lebih konsisten (mengganti spasi dengan underscore, huruf kecil)
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

    # Menghapus kolom yang tidak relevan
    df_cleaned = df.drop(['id', 'delivery_person_id', 'order_date', 'time_orderd', 'time_order_picked'], axis=1)

    # Menghitung jarak menggunakan haversine
    df_cleaned['distance'] = df_cleaned.apply(
        lambda row: haversine(
            (row['restaurant_latitude'], row['restaurant_longitude']),
            (row['delivery_location_latitude'], row['delivery_location_longitude']),
            unit=Unit.KILOMETERS
        ), axis=1
    )
    df_cleaned = df_cleaned.drop(['restaurant_latitude', 'restaurant_longitude', 'delivery_location_latitude', 'delivery_location_longitude'], axis=1)

    # Mengatasi nilai NaN dan mereset index
    df_cleaned.dropna(inplace=True)

    # Mengubah tipe data kolom yang salah
    for col in ['delivery_person_age', 'delivery_person_ratings', 'multiple_deliveries']:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    df_cleaned.dropna(inplace=True)

    # 3. Persiapan untuk Model Training
    # **FIX**: Menggunakan nama kolom yang sudah dibersihkan 'time_taken_(min)'
    X = df_cleaned.drop('time_taken_(min)', axis=1)
    y = df_cleaned['time_taken_(min)']

    # Identifikasi kolom numerik dan kategorikal
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # 4. Membuat Pipeline Preprocessing dan Model
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42))
    ])

    # 5. Melatih Model
    model_pipeline.fit(X, y)
    
    return model_pipeline, df_cleaned

# --- Memuat model dan data (hanya berjalan sekali) ---
try:
    final_model, df_for_ui = load_and_train_model('Zomato.csv')
except FileNotFoundError:
    st.error("File 'Zomato.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses data atau melatih model: {e}")
    st.stop()

# =============================================================================
# --- UI (User Interface) Aplikasi Streamlit ---
# =============================================================================

st.title('🛵 Prediksi Waktu Pengiriman Makanan Zomato')
st.markdown("Aplikasi ini menggunakan *Machine Learning* untuk memprediksi estimasi waktu pengiriman. Silakan masukkan parameter di sidebar kiri.")

# --- Sidebar untuk Input dari Pengguna ---
st.sidebar.header('Masukkan Parameter Pengiriman')

def get_user_inputs():
    """Membuat sidebar untuk semua input pengguna."""
    age = st.sidebar.slider('Usia Pengantar', 
                            int(df_for_ui['delivery_person_age'].min()), 
                            int(df_for_ui['delivery_person_age'].max()), 
                            25)
                            
    ratings = st.sidebar.slider('Rating Pengantar', 
                                float(df_for_ui['delivery_person_ratings'].min()), 
                                float(df_for_ui['delivery_person_ratings'].max()), 
                                4.5, 0.1)
                                
    distance = st.sidebar.number_input('Jarak Pengiriman (km)', 
                                       min_value=0.1, 
                                       max_value=50.0, 
                                       value=5.0, 
                                       step=0.5)

    st.sidebar.markdown("---")
    
    # **FIX**: Menggunakan nama kolom yang sudah dibersihkan
    weather = st.sidebar.selectbox('Kondisi Cuaca', sorted(df_for_ui['weather_conditions'].unique()))
    traffic = st.sidebar.selectbox('Kepadatan Lalu Lintas', sorted(df_for_ui['road_traffic_density'].unique()))
    vehicle_type = st.sidebar.selectbox('Tipe Kendaraan', sorted(df_for_ui['type_of_vehicle'].unique()))
    order_type = st.sidebar.selectbox('Tipe Pesanan', sorted(df_for_ui['type_of_order'].unique()))

    st.sidebar.markdown("---")

    vehicle_cond = st.sidebar.slider('Kondisi Kendaraan (0=Buruk, 3=Sangat Baik)', 
                                     int(df_for_ui['vehicle_condition'].min()), 
                                     int(df_for_ui['vehicle_condition'].max()), 
                                     2)
                                     
    multiple_del = st.sidebar.slider('Jumlah Pengiriman Bersamaan', 
                                     int(df_for_ui['multiple_deliveries'].min()), 
                                     int(df_for_ui['multiple_deliveries'].max()), 
                                     1)
                                     
    festival = st.sidebar.selectbox('Apakah Sedang Festival?', sorted(df_for_ui['festival'].unique()))
    city = st.sidebar.selectbox('Tipe Kota', sorted(df_for_ui['city'].unique()))

    # Membuat DataFrame dari input
    input_data = {
        'delivery_person_age': age,
        'delivery_person_ratings': ratings,
        'weather_conditions': weather,
        'road_traffic_density': traffic,
        'vehicle_condition': vehicle_cond,
        'type_of_order': order_type,
        'type_of_vehicle': vehicle_type,
        'multiple_deliveries': multiple_del,
        'festival': festival,
        'city': city,
        'distance': distance
    }
    input_df = pd.DataFrame([input_data])
    return input_df

user_input_df = get_user_inputs()

# --- Tampilan Utama ---
st.divider()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header('Parameter yang Anda Pilih')
    st.dataframe(user_input_df.T.rename(columns={0: 'Values'})) # Transpose untuk tampilan lebih baik

with col2:
    st.header('Hasil Prediksi')
    predict_button = st.button('**Hitung Estimasi Waktu**', use_container_width=True, type="primary")
    
    if predict_button:
        try:
            prediction = final_model.predict(user_input_df)
            predicted_time = int(round(prediction[0]))

            st.success(f"Pesanan Anda diperkirakan akan tiba dalam **{predicted_time} menit**.", icon="✅")
            
            st.metric(label="Estimasi Waktu Pengiriman", value=f"{predicted_time} menit")

            # Penjelasan singkat
            st.caption("Prediksi ini didasarkan pada model Machine Learning yang telah dilatih pada data historis Zomato.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
