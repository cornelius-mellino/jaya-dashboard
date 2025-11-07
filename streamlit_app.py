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

# Kolom Fitur untuk Model (harus sama dengan yang digunakan saat pelatihan)
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
    df['Target_Biner'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})
    
    # 3. Seleksi Fitur (Hapus fitur 2nd sem dan kolom target asli)
    df = df.drop('Target', axis=1)
    if 'Application order' in df.columns:
        df = df.drop('Application order', axis=1)
    cols_to_drop = [col for col in df.columns if '2nd sem' in col]
    df = df.drop(cols_to_drop, axis=1)
    
    X = df.drop('Target_Biner', axis=1)
    y = df['Target_Biner']
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, X, y

# Fungsi untuk melatih model dan mendapatkan pipeline (Gunakan cache)
@st.cache_resource
def train_model(X_train, y_train):
    # Pipeline untuk data numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk data kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Pipeline untuk data biner
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # Gabungkan semua transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bin', binary_transformer, BINARY_FEATURES)
        ],
        remainder='drop'
    )
    
    # Buat instance model Random Forest
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
    ohe_feature_names = preprocessor.named_transformers_['cat'] \
                        .named_steps['onehot'] \
                        .get_feature_names_out(CATEGORICAL_FEATURES)
    
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
st.title("🎓 Dashboard Deteksi Dropout Dini Mahasiswa")
st.markdown("Aplikasi berbasis Model Random Forest untuk memprediksi risiko *dropout* setelah Semester 1.")

# Memuat dan melatih model
try:
    X_train, X_test, y_train, y_test, X_full, y_full = load_and_prepare_data(FILE_PATH)
    model_pipeline = train_model(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilitas Dropout
except Exception as e:
    st.error(f"Gagal memuat atau melatih model. Pastikan file '{FILE_PATH}' ada di direktori yang sama. Error: {e}")
    st.stop()


# ==================================================
# Bagian 1: Model Performance & Metrik
# ==================================================
st.header("1. Kinerja Model Random Forest")

col1, col2, col3 = st.columns(3)

# Akurasi
accuracy = model_pipeline.score(X_test, y_test)
col1.metric("Akurasi Model", f"{accuracy:.2%}")

# Presisi & Recall (Fokus pada kelas Dropout/1)
report = classification_report(y_test, y_pred, output_dict=True)

dropout_metrics = report['1'] # Fix: Use '1' as key for the Dropout class
col2.metric("Presisi (Menduga Dropout)", f"{dropout_metrics['precision']:.2f}")
col3.metric("Recall (Mengidentifikasi Dropout)", f"{dropout_metrics['recall']:.2f}")

st.markdown("""
<div style="border-left: 5px solid #ff4b4b; padding: 10px; margin-bottom: 20px;">
    <b>Interpretasi Kinerja:</b>
    <ul>
        <li><b>Presisi:</b> Dari semua siswa yang diprediksi <b>Dropout</b>, <code>{:.0f}%</code> benar-benar <b>Dropout</b>.</li>
        <li><b>Recall:</b> Dari semua siswa yang sebenarnya <b>Dropout</b>, model berhasil mengidentifikasi <code>{:.0f}%</code> di antaranya.</li>
    </ul>
</div>
""".format(dropout_metrics['precision']*100, dropout_metrics['recall']*100), unsafe_allow_html=True)

# Confusion Matrix
st.subheader("Visualisasi Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=['Tidak Dropout', 'Dropout'],
    cmap='Blues',
    ax=ax
)
ax.set_title('Confusion Matrix Model Random Forest')
st.pyplot(fig)

# ==================================================
# Bagian 2: Feature Importance (Wawasan Bisnis)
# ==================================================
st.header("2. Faktor Pendorong Utama Dropout")
st.markdown("Fitur-fitur ini menunjukkan variabel mana yang paling kuat memengaruhi keputusan model untuk memprediksi *dropout*.")

feature_importance_df = get_feature_importance(model_pipeline, X_full)

# Visualisasi Feature Importance
top_n = st.slider("Pilih Jumlah Fitur Teratas untuk Ditampilkan:", 5, 20, 15)

fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
sns.barplot(
    data=feature_importance_df.head(top_n), 
    x='Importance', 
    y='Feature', 
    palette='viridis',
    ax=ax_imp
)
ax_imp.set_title(f'Top {top_n} Fitur Terpenting untuk Memprediksi Dropout')
ax_imp.set_xlabel('Tingkat Kepentingan (Gini Importance)')
ax_imp.set_ylabel('Fitur')
plt.tight_layout()
st.pyplot(fig_imp)

st.subheader("📝 Actionable Insights dari Top 5 Fitur:")
top_5 = feature_importance_df.head(5)['Feature'].tolist()
st.markdown(f"""
* **1. `{top_5[0]}`:** Faktor akademik ini adalah penentu terkuat. Intervensi harus fokus pada peningkatan nilai S1.
* **2. `{top_5[1]}`:** Keberhasilan menyelesaikan unit S1 adalah indikator utama. Mahasiswa yang lulus kurang dari 3 unit (dari EDA) berisiko tinggi.
* **3. `{top_5[2]}`:** Status pembayaran UKT adalah sinyal finansial krisis yang harus diprioritaskan untuk bantuan.
* **4. `{top_5[3]}`:** Menariknya, Nilai Masuk memiliki pengaruh yang lebih kecil daripada performa aktual S1.
* **5. `{top_5[4]}`:** Penerima Beasiswa jauh lebih kecil kemungkinannya untuk *dropout*. Ini mendukung investasi dalam program beasiswa.
""")

def get_feature_importance(model_pipeline, X_full):
    """
    Extracts feature importance from a pipeline containing a preprocessor and a classifier.

    Args:
        model_pipeline: A scikit-learn Pipeline object.
        X_full: The original DataFrame of features (before train/test split).

    Returns:
        A pandas DataFrame with 'Feature' names and their 'Importance',
        sorted by importance in descending order.
    """
    # Get the trained classifier from the pipeline
    classifier = model_pipeline.named_steps['classifier']

    # Get the preprocessor from the pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']

    # Get the feature names after preprocessing
    # This method is available in ColumnTransformer after fitting
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        print(f"Could not get feature names from preprocessor: {e}")
        # Fallback: try to get feature names from the original columns and OHE names
        # This requires knowing the structure of the preprocessor
        try:
            numeric_features_processed = preprocessor.named_transformers_['num'].get_feature_names_out() # Scaled numerical features
            categorical_features_processed = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(X_full.columns[preprocessor.transformers_[1][2]]) # One-hot encoded categorical features
            binary_features_processed = preprocessor.named_transformers_['bin'].get_feature_names_out() # Binary features

            # Combine feature names in the correct order
            feature_names = list(numeric_features_processed) + list(categorical_features_processed) + list(binary_features_processed)

            # Filter out columns that were dropped by the preprocessor
            # This is a heuristic and might not be perfect
            original_cols_processed = preprocessor.transform(X_full).shape[1]
            if len(feature_names) != original_cols_processed:
                print("Warning: Mismatch between generated feature names and preprocessor output shape.")
                # As a last resort, create generic names if mismatch occurs
                feature_names = [f'feature_{i}' for i in range(original_cols_processed)]


        except Exception as e_fallback:
            print(f"Fallback feature name generation failed: {e_fallback}")
            # If all attempts fail, use generic names
            feature_names = [f'feature_{i}' for i in range(preprocessor.transform(X_full).shape[1])]
            print("Using generic feature names.")


    # Get feature importances from the classifier
    importances = classifier.feature_importances_

    # Create a DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return feature_importance_df

# ==================================================
# Bagian 3: Prediksi Interaktif (Risk Checker)
# ==================================================
st.header("3. Prediksi Risiko Individual (Risk Checker)")
st.sidebar.header("Input Data Mahasiswa S1")

# Sidebar Input: Fitur Akademik
st.sidebar.subheader("Data Akademik Semester 1")
input_grade = st.sidebar.number_input(
    'Rata-rata Nilai S1 (Curricular units 1st sem (grade))', 
    min_value=0.0, max_value=20.0, value=10.0, step=0.1
)
input_approved = st.sidebar.number_input(
    'Unit Kurikuler Lulus S1 (Curricular units 1st sem (approved))', 
    min_value=0, max_value=20, value=3
)
input_admission_grade = st.sidebar.number_input(
    'Nilai Masuk (Admission grade)', 
    min_value=95.0, max_value=190.0, value=127.0, step=0.1
)

# Sidebar Input: Fitur Finansial
st.sidebar.subheader("Status Finansial/Sosial")
input_fees = st.sidebar.selectbox(
    'UKT Lancar? (Tuition fees up to date)', 
    options=[1, 0], format_func=lambda x: 'Ya (1)' if x == 1 else 'Tidak (0)'
)
input_scholarship = st.sidebar.selectbox(
    'Penerima Beasiswa? (Scholarship holder)', 
    options=[0, 1], format_func=lambda x: 'Ya (1)' if x == 1 else 'Tidak (0)'
)
input_debtor = st.sidebar.selectbox(
    'Memiliki Hutang? (Debtor)', 
    options=[0, 1], format_func=lambda x: 'Ya (1)' if x == 1 else 'Tidak (0)'
)
input_age = st.sidebar.number_input('Usia Saat Masuk (Age at enrollment)', min_value=17, max_value=70, value=20)


# --- Membuat Input DataFrame ---
# Membuat DataFrame tunggal untuk prediksi dengan nilai default untuk semua kolom
# Kolom yang tidak diinput user akan diisi dengan median/modus dari data latih (X_train)
input_data = X_train.head(1).copy() 
input_data.iloc[0] = X_train.mode().iloc[0] # Isi dengan modus/median data latih

# Override dengan Input User
input_data.loc[:, 'Curricular units 1st sem (grade)'] = input_grade
input_data.loc[:, 'Curricular units 1st sem (approved)'] = input_approved
input_data.loc[:, 'Admission grade'] = input_admission_grade
input_data.loc[:, 'Tuition fees up to date'] = input_fees
input_data.loc[:, 'Scholarship holder'] = input_scholarship
input_data.loc[:, 'Debtor'] = input_debtor
input_data.loc[:, 'Age at enrollment'] = input_age


# Lakukan Prediksi
if st.sidebar.button('Hitung Risiko Dropout'):
    # Prediksi Probabilitas
    risk_proba = model_pipeline.predict_proba(input_data)[0][1] * 100
    
    st.subheader("Hasil Prediksi Risiko")
    
    col_risk1, col_risk2 = st.columns(2)
    
    col_risk1.metric(
        "Probabilitas Dropout", 
        f"{risk_proba:.2f}%"
    )

    if risk_proba >= 50:
        risk_level = "SANGAT TINGGI"
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
        Tingkat Risiko: {risk_level}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    if risk_proba >= 30:
        st.warning("⚠️ **Rekomendasi Intervensi:** Mahasiswa ini memerlukan konseling akademik segera dan evaluasi status finansial. Fokus pada unit-unit yang gagal di S1.")
    else:
        st.success("✅ **Rekomendasi:** Risiko rendah. Tetap pantau perkembangan di semester selanjutnya.")
