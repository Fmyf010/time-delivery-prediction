import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
# Menggunakan cache_resource agar proses ini hanya berjalan sekali saat aplikasi pertama kali dimuat.
# Ini akan membuat aplikasi jauh lebih cepat setelah pemuatan awal.
@st.cache_resource
def load_and_train_model(data_path):
    """
    Fungsi ini memuat data, membersihkannya, melakukan feature engineering,
    dan melatih model machine learning.
    """
    # 1. Memuat Data
    df = pd.read_csv(data_path)

    # 2. Pembersihan Data dan Feature Engineering
    # Menghapus spasi ekstra dari nama kolom
    df.columns = df.columns.str.strip()
    
    # Menghapus kolom yang tidak relevan
    df_cleaned = df.drop(['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked'], axis=1)

    # Menghitung jarak menggunakan haversine
    df_cleaned['distance'] = df_cleaned.apply(
        lambda row: haversine(
            (row['Restaurant_latitude'], row['Restaurant_longitude']),
            (row['Delivery_location_latitude'], row['Delivery_location_longitude']),
            unit=Unit.KILOMETERS
        ), axis=1
    )
    # Menghapus kolom latitude dan longitude setelah perhitungan jarak
    df_cleaned = df_cleaned.drop(['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude'], axis=1)

    # Menghapus nilai NaN dan mereset index
    df_cleaned.dropna(inplace=True)

    # Mengubah tipe data kolom yang salah
    df_cleaned['Delivery_person_Age'] = pd.to_numeric(df_cleaned['Delivery_person_Age'], errors='coerce')
    df_cleaned['Delivery_person_Ratings'] = pd.to_numeric(df_cleaned['Delivery_person_Ratings'], errors='coerce')
    df_cleaned['multiple_deliveries'] = pd.to_numeric(df_cleaned['multiple_deliveries'], errors='coerce')
    df_cleaned.dropna(inplace=True) # Hapus NaN lagi setelah konversi tipe data

    # 3. Persiapan untuk Model Training
    X = df_cleaned.drop('Time_taken(min)', axis=1)
    y = df_cleaned['Time_taken(min)']

    # Identifikasi kolom numerik dan kategorikal
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # 4. Membuat Pipeline Preprocessing dan Model
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Membuat pipeline lengkap dengan model LightGBM
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
    st.stop() # Menghentikan eksekusi jika file tidak ada

# =============================================================================
# --- UI (User Interface) Aplikasi Streamlit ---
# =============================================================================

st.title('🛵 Prediksi Waktu Pengiriman Makanan Zomato')
st.markdown("Aplikasi ini menggunakan *Machine Learning* untuk memprediksi estimasi waktu pengiriman. Silakan masukkan parameter di sidebar kiri.")
st.markdown("---")

# --- Sidebar untuk Input dari Pengguna ---
st.sidebar.header('Masukkan Parameter Pengiriman')

def get_user_inputs():
    """Membuat sidebar untuk semua input pengguna."""
    age = st.sidebar.slider('Usia Pengantar', 
                            int(df_for_ui['Delivery_person_Age'].min()), 
                            int(df_for_ui['Delivery_person_Age'].max()), 
                            25)
                            
    ratings = st.sidebar.slider('Rating Pengantar', 
                                float(df_for_ui['Delivery_person_Ratings'].min()), 
                                float(df_for_ui['Delivery_person_Ratings'].max()), 
                                4.5)
                                
    distance = st.sidebar.number_input('Jarak Pengiriman (km)', 
                                       min_value=0.1, 
                                       max_value=50.0, 
                                       value=5.0, 
                                       step=0.5)

    st.sidebar.markdown("---")

    weather = st.sidebar.selectbox('Kondisi Cuaca', sorted(df_for_ui['Weather_conditions'].unique()))
    traffic = st.sidebar.selectbox('Kepadatan Lalu Lintas', sorted(df_for_ui['Road_traffic_density'].unique()))
    vehicle_type = st.sidebar.selectbox('Tipe Kendaraan', sorted(df_for_ui['Type_of_vehicle'].unique()))
    order_type = st.sidebar.selectbox('Tipe Pesanan', sorted(df_for_ui['Type_of_order'].unique()))

    st.sidebar.markdown("---")

    vehicle_cond = st.sidebar.slider('Kondisi Kendaraan (0=Buruk, 3=Sangat Baik)', 
                                     int(df_for_ui['Vehicle_condition'].min()), 
                                     int(df_for_ui['Vehicle_condition'].max()), 
                                     2)
                                     
    multiple_del = st.sidebar.slider('Jumlah Pengiriman Bersamaan', 
                                     int(df_for_ui['multiple_deliveries'].min()), 
                                     int(df_for_ui['multiple_deliveries'].max()), 
                                     1)
                                     
    festival = st.sidebar.selectbox('Apakah Sedang Festival?', sorted(df_for_ui['Festival'].unique()))
    city = st.sidebar.selectbox('Tipe Kota', sorted(df_for_ui['City'].unique()))

    # Membuat DataFrame dari input
    input_data = {
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': ratings,
        'Weather_conditions': weather,
        'Road_traffic_density': traffic,
        'Vehicle_condition': vehicle_cond,
        'Type_of_order': order_type,
        'Type_of_vehicle': vehicle_type,
        'multiple_deliveries': multiple_del,
        'Festival': festival,
        'City': city,
        'distance': distance
    }
    input_df = pd.DataFrame([input_data])
    return input_df

# Mendapatkan input dari pengguna
user_input_df = get_user_inputs()

# Menampilkan input pengguna di halaman utama
st.header('Parameter yang Anda Pilih')
st.dataframe(user_input_df, use_container_width=True)

# Tombol untuk melakukan prediksi
st.markdown("---")
predict_button = st.button('**Prediksi Waktu Pengiriman**', use_container_width=True, type="primary")

# --- Logika Prediksi dan Tampilan Hasil ---
if predict_button:
    try:
        # Melakukan prediksi menggunakan model yang sudah dilatih
        prediction = final_model.predict(user_input_df)
        predicted_time = int(round(prediction[0]))

        # Menampilkan hasil prediksi dengan desain yang menarik
        st.subheader('⭐ Hasil Prediksi')
        st.success(f"**Estimasi Waktu Pengiriman: {predicted_time} menit**", icon="🎉")
        
        # Menambahkan visualisasi atau metrik untuk konteks
        col1, col2, col3 = st.columns(3)
        col1.metric("Jarak", f"{user_input_df['distance'].iloc[0]} km")
        col2.metric("Lalu Lintas", user_input_df['Road_traffic_density'].iloc[0])
        col3.metric("Cuaca", user_input_df['Weather_conditions'].iloc[0])

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
