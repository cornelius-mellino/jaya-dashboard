import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ==============================================================================
# 1. KONFIGURASI DAN UTILITAS (Caching untuk performa)
# ==============================================================================

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard Deteksi Dropout Mahasiswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tentukan nama file
FILE_PATH = 'data.csv'

# Daftar Fitur (Sama seperti yang digunakan saat pelatihan)
NUMERIC_FEATURES = [
    'Admission grade', 'Previous qualification (grade)', 'Age at enrollment',
    'Unemployment rate', 'Inflation rate', 'GDP',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)'
]

CATEGORICAL_FEATURES = [
    'Marital status', 'Application mode', 'Course', 
    'Previous qualification', 'Nacionality', 
    "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation"
]

BINARY_FEATURES = [
    'Daytime/evening attendance\t', 'Displaced', 'Educational special needs', 
    'Debtor', 'Tuition fees up to date', 'Gender', 
    'Scholarship holder', 'International'
]

# Fungsi untuk memuat data dan melakukan preprocessing dasar (Gunakan cache)
@st.cache_data
def load_and_prepare_data(file_path):
    # 1. Muat Data
    df = pd.read_csv(file_path, delimiter=';')
    
    # 2. Pembersihan & Feature Engineering
    df.drop_duplicates(inplace=True)
    # Target 1=Dropout, 0=Not Dropout (Graduate/Enrolled)
    df['Target_Biner'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})
    
    # 3. Seleksi Fitur (Hapus fitur 2nd sem dan kolom target asli)
    target_original = df['Target'].copy() # Simpan target original untuk EDA
    df = df.drop('Target', axis=1)
    if 'Application order' in df.columns:
        df = df.drop('Application order', axis=1)
    # Hapus kolom semester 2 (pencegahan data leakage)
    cols_to_drop = [col for col in df.columns if '2nd sem' in col]
    df = df.drop(cols_to_drop, axis=1)
    
    X = df.drop('Target_Biner', axis=1)
    y = df['Target_Biner']
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, X, y, target_original

# Fungsi untuk melatih model dan mendapatkan pipeline (Gunakan cache)
@st.cache_resource
def train_model(X_train, y_train):
    # Pipeline Preprocessing (Imputasi, Scaling, Encoding)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bin', binary_transformer, BINARY_FEATURES)
        ],
        remainder='drop'
    )
    
    # Model Random Forest dengan class_weight='balanced'
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Gabungkan preprocessor dan model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_model)
    ])
    
    # Latih model
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

# Fungsi untuk mendapatkan Feature Importance
def get_feature_importance(model_pipeline, X):
    rf_model = model_pipeline.named_steps['classifier']
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Dapatkan nama fitur setelah OHE
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'] \
                            .named_steps['onehot'] \
                            .get_feature_names_out(CATEGORICAL_FEATURES)
    except AttributeError:
        ohe_feature_names = [f'Cat_{i}' for i in range(X.shape[1] - len(NUMERIC_FEATURES) - len(BINARY_FEATURES))]
        
    all_feature_names = NUMERIC_FEATURES + list(ohe_feature_names) + BINARY_FEATURES
    
    # Dapatkan nilai importance
    importances = rf_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance_df

# ==============================================================================
# 2. TAMPILAN DASHBOARD
# ==============================================================================

# Header
st.title("🎓 Dashboard Deteksi Risiko Dropout Dini Mahasiswa")
st.markdown("Aplikasi berbasis **Model Random Forest** untuk memprediksi risiko *dropout* setelah Semester 1 (S1).")

# Memuat dan melatih model
try:
    X_train, X_test, y_train, y_test, X_full, y_full, target_original = load_and_prepare_data(FILE_PATH)
    # Gabungkan kembali X_full dan target_original untuk EDA
    df_eda = pd.concat([X_full.reset_index(drop=True), target_original.reset_index(drop=True)], axis=1)
    
    model_pipeline = train_model(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    # Ambil metrik performa awal
    report = classification_report(y_test, y_pred, output_dict=True)
    dropout_metrics = report['1'] # Kelas 1 adalah Dropout
    accuracy = model_pipeline.score(X_test, y_test)
    
except FileNotFoundError:
    st.error(f"⚠️ **Error:** File '{FILE_PATH}' tidak ditemukan. Mohon pastikan file CSV berada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Gagal memuat atau melatih model.** Pastikan format data (delimiter: ';') sudah benar. Error: {e}")
    st.stop()

st.sidebar.header("⚙️ Konfigurasi & Intervensi")
st.sidebar.markdown("Gunakan bagian ini untuk memprediksi risiko seorang mahasiswa.")

# ==================================================
# Bagian 1: Model Performance & Metrik
# ==================================================
st.header("1. Kinerja Model Random Forest")

col1, col2, col3 = st.columns(3)

# Akurasi
col1.metric("Akurasi Keseluruhan", f"{accuracy:.2%}")

# Presisi & Recall (Fokus pada kelas Dropout/1)
col2.metric("Presisi (Menduga Dropout)", f"{dropout_metrics['precision']:.2f}", help="Tingkat kebenaran prediksi 'Dropout'.")
col3.metric("Recall (Mengidentifikasi Dropout)", f"{dropout_metrics['recall']:.2f}", help="Tingkat keberhasilan model menangkap kasus 'Dropout' yang sebenarnya.")

st.markdown(
    """
    <div style="border-left: 5px solid #007bff; padding: 10px; margin-bottom: 20px; background-color: #f0f8ff;">
        <b>Tujuan Metrik:</b> Untuk deteksi dini, <b>Recall</b> yang tinggi pada kelas 'Dropout' sangat penting untuk intervensi yang berhasil.
    </div>
    """, unsafe_allow_html=True
)

# ==================================================
# Bagian 2: Exploratory Data Analysis (EDA) Charts
# ==================================================
st.header("2. Exploratory Data Analysis (EDA) Kunci")
st.markdown("Analisis visual dari data mentah untuk memahami faktor pendorong risiko.")

tab1, tab2 = st.tabs(["Distribusi & Finansial", "Performa Akademik"])

with tab1:
    col_dist, col_fin = st.columns(2)
    
    # Grafik 1: Distribusi Target (Count Plot)
    with col_dist:
        st.subheader("Distribusi Status Mahasiswa")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df_eda, x='Target', order=df_eda['Target'].value_counts().index, palette='viridis', ax=ax)
        ax.set_title('Status Akhir Mahasiswa')
        ax.set_xlabel('Status Akhir')
        ax.set_ylabel('Jumlah Mahasiswa')
        st.pyplot(fig)
    
    # Grafik 4: UKT vs. Target (Count Plot)
    with col_fin:
        st.subheader("Status UKT vs. Risiko Dropout")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df_eda, x='Tuition fees up to date', hue='Target', palette='magma', ax=ax)
        ax.set_title('Kelancaran UKT vs. Status')
        ax.set_xlabel('UKT Lancar (1=Ya, 0=Tidak)')
        ax.set_ylabel('Jumlah Mahasiswa')
        ax.legend(title='Status Akhir')
        st.pyplot(fig)
        st.caption("Fokus: Proporsi Dropout yang sangat tinggi ketika UKT 'Tidak Lancar' (0).")


with tab2:
    col_grade, col_approved = st.columns(2)
    
    # Grafik 7: Nilai Semester 1 vs. Target (Box Plot)
    with col_grade:
        st.subheader("Rata-rata Nilai Semester 1")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df_eda, x='Target', y='Curricular units 1st sem (grade)', palette='viridis', ax=ax)
        ax.set_title('Nilai S1 vs. Status')
        ax.set_xlabel('Status Akhir')
        ax.set_ylabel('Nilai S1')
        st.pyplot(fig)
        st.caption("Fokus: Median Nilai Dropout jauh lebih rendah dibandingkan Graduate/Enrolled.")

    # Grafik 9: Unit Lulus Semester 1 vs. Target (Box Plot)
    with col_approved:
        st.subheader("Jumlah Unit Lulus Semester 1")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df_eda, x='Target', y='Curricular units 1st sem (approved)', palette='viridis', ax=ax)
        ax.set_title('Unit Lulus S1 vs. Status')
        ax.set_xlabel('Status Akhir')
        ax.set_ylabel('Unit Lulus S1')
        st.pyplot(fig)
        st.caption("Fokus: Mahasiswa Dropout cenderung lulus hanya 2-3 unit, yang lulus bisa 5-6 unit.")


# ==================================================
# Bagian 3: Feature Importance (Wawasan Bisnis)
# ==================================================
st.header("3. Faktor Pendorong Utama Dropout (Model)")
feature_importance_df = get_feature_importance(model_pipeline, X_full)

col_imp, col_insight = st.columns([2, 1])

# Visualisasi Feature Importance
with col_imp:
    top_n = st.slider("Pilih Jumlah Fitur Teratas:", 5, 20, 10)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=feature_importance_df.head(top_n), 
        x='Importance', 
        y='Feature', 
        palette='magma',
        ax=ax_imp
    )
    ax_imp.set_title(f'Top {top_n} Fitur Terpenting (Sinyal Dini)')
    ax_imp.set_xlabel('Tingkat Kepentingan')
    ax_imp.set_ylabel('Fitur')
    plt.tight_layout()
    st.pyplot(fig_imp)

# Actionable Insights di kolom terpisah
with col_insight:
    st.subheader("Actionable Summary")
    st.markdown("""
    Model menekankan bahwa **performa akademik** di S1 lebih penting daripada nilai masuk atau faktor sosio-ekonomi lainnya.
    
    1.  **Prioritas Tertinggi (Akademik):** Rata-rata Nilai S1 (`Curricular units 1st sem (grade)`) dan Jumlah Unit Lulus S1 (`Curricular units 1st sem (approved)`).
    2.  **Sinyal Cepat (Finansial):** Kelancaran UKT (`Tuition fees up to date`) adalah *flag* risiko yang paling mudah didapat.
    3.  **Potensi *Bias*:** Perluasan beasiswa (`Scholarship holder`) adalah intervensi yang terbukti efektif.
    """)


# ==================================================
# Bagian 4: Prediksi Interaktif (Risk Checker)
# ==================================================
st.header("4. Prediksi Risiko Individual")
st.markdown("Gunakan panel di *sidebar* untuk menyesuaikan input mahasiswa dan memprediksi risikonya.")

# Input untuk Sidebar (diulang untuk memastikan kode lengkap)
st.sidebar.markdown("---")
st.sidebar.subheader("Input Data Mahasiswa S1")

# Input Akademik
st.sidebar.caption("Sinyal Akademik S1")
input_grade = st.sidebar.slider(
    'Rata-rata Nilai S1', 
    min_value=0.0, max_value=20.0, value=10.0, step=0.1
)
input_approved = st.sidebar.slider(
    'Unit Kurikuler Lulus S1', 
    min_value=0, max_value=10, value=3
)
input_admission_grade = st.sidebar.slider(
    'Nilai Masuk', 
    min_value=95.0, max_value=190.0, value=127.0, step=1.0
)

# Input Finansial
st.sidebar.caption("Sinyal Finansial")
input_fees = st.sidebar.selectbox(
    'UKT Lancar? (1=Ya, 0=Tidak)', 
    options=[1, 0], format_func=lambda x: 'Ya (1)' if x == 1 else 'Tidak (0)'
)
input_scholarship = st.sidebar.selectbox(
    'Penerima Beasiswa? (1=Ya, 0=Tidak)', 
    options=[0, 1], format_func=lambda x: 'Ya (1)' if x == 1 else 'Tidak (0)'
)

# Input Demografis
st.sidebar.caption("Sinyal Demografis")
input_age = st.sidebar.number_input('Usia Saat Masuk', min_value=17, max_value=70, value=20)


# --- Membuat Input DataFrame ---
if 'input_data' not in st.session_state:
    st.session_state.input_data = X_train.head(1).copy() 
    for col in st.session_state.input_data.columns:
        if col in NUMERIC_FEATURES:
            st.session_state.input_data.loc[:, col] = X_train[col].median()
        else:
            st.session_state.input_data.loc[:, col] = X_train[col].mode().iloc[0]

current_input_data = st.session_state.input_data.copy()
current_input_data.loc[:, 'Curricular units 1st sem (grade)'] = input_grade
current_input_data.loc[:, 'Curricular units 1st sem (approved)'] = input_approved
current_input_data.loc[:, 'Admission grade'] = input_admission_grade
current_input_data.loc[:, 'Tuition fees up to date'] = input_fees
current_input_data.loc[:, 'Scholarship holder'] = input_scholarship
current_input_data.loc[:, 'Age at enrollment'] = input_age
current_input_data.loc[:, 'Debtor'] = 0 # Defaultkan Debtor menjadi 0


if st.button('Hitung Risiko Mahasiswa'):
    with st.spinner('Menghitung risiko...'):
        risk_proba = model_pipeline.predict_proba(current_input_data)[0][1] * 100
        
        st.subheader("Hasil Prediksi Risiko")
        
        col_risk1, col_risk2 = st.columns(2)
        
        col_risk1.metric(
            "Probabilitas Dropout", 
            f"{risk_proba:.2f}%"
        )

        if risk_proba >= 50:
            risk_level = "KRITIS"
            color = "#ff4b4b"
        elif risk_proba >= 30:
            risk_level = "TINGGI"
            color = "#ffa500"
        else:
            risk_level = "RENDAH"
            color = "#008000"

        col_risk2.markdown(
            f"""
            <div style="
                background-color: {color}; 
                color: white; 
                padding: 15px; 
                border-radius: 5px; 
                text-align: center;
                font-size: 18px;
                font-weight: bold;
            ">
            Tingkat Risiko: {emoji} {risk_level}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        if risk_proba >= 30:
            st.error("❗ **TINDAKAN SEGERA DIPERLUKAN:** Mahasiswa ini memerlukan intervensi terfokus. Kontak segera dengan konselor akademis dan unit bantuan keuangan.")
        else:
            st.info("👍 **TINDAKAN PANTUAN:** Risiko rendah. Tetap pantau secara berkala.")