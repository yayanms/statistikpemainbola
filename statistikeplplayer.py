import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ==========================
# TITLE
# ==========================
st.title("📊 Sistem Analisis Statistik Pemain Bola")

st.write("Gunakan dataset default atau upload dataset sendiri untuk memulai analisis lengkap.")

# ==========================
# PILIH SUMBER DATA
# ==========================
st.subheader("📁 Pilih Sumber Data")

pilihan_data = st.radio(
    "Pilih cara memasukkan data:",
    ("Gunakan Dataset Default", "Upload Dataset Sendiri")
)

df = None  # siapkan variabel

# ==========================
# DATASET DEFAULT
# ==========================
if pilihan_data == "Gunakan Dataset Default":
    st.info("Menggunakan dataset default bawaan aplikasi.")
    
    try:
        df = pd.read_csv("fbref_PL_2024-25.csv")  # ganti nama file sesuai file Anda
    except:
        st.error("❌ Tidak menemukan file default. Pastikan file 'dataset_default.csv' berada 1 folder dengan script.")
        st.stop()

# ==========================
# UPLOAD DATASET SENDIRI
# ==========================
if pilihan_data == "Upload Dataset Sendiri":
    uploaded_file = st.file_uploader("Upload file CSV Anda:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Silakan upload file CSV terlebih dahulu.")
        st.stop()

# ==========================
# PROSES DATA
# ==========================
# RENAME KOLOM
rename_dict = {
    "Player": "Nama Pemain",
    "Squad": "Klub",
    "Pos": "Posisi",
    "Age": "Usia",
    "Gls": "Goal",
    "Ast": "Assist",
    "CrdY": "Kartu Kuning",
    "CrdR": "Kartu Merah",
    "Min": "Menit Bermain",
    "Sh": "Tembakan",
    "SoT": "Tembakan Tepat Sasaran",
    "Fls": "Pelanggaran",
    "xG": "Expected Goals",
    "xAG": "Expected Assist",
    "Touches": "Sentuhan Bola",
    "Tkl": "Tackle",
    "Int": "Intersepsi",
    "Clr": "Sapuan",
    "Blocks": "Blok",
    "Cmp": "Operan Berhasil",
    "Att": "Operan Dicoba",
    "PrgP": "Operan Progresif",
    "PrgC": "Carry Progresif",
    "PrgR": "Lari Progresif",
}

df.rename(columns=rename_dict, inplace=True)

st.subheader("📘 Data Dengan Nama Kolom Mudah Dipahami")
st.dataframe(df.head())

# ==========================
# CLEANING + SCALING
# ==========================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_clean = imputer.fit_transform(df[numeric_cols])
X_scaled = scaler.fit_transform(X_clean)

# ==========================
# PCA
# ==========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["Komponen Utama 1", "Komponen Utama 2"])

# ==========================
# CLUSTERING
# ==========================
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)
pca_df["Cluster"] = labels

st.subheader("📌 Hasil PCA + Cluster")
st.dataframe(pca_df)

# ==========================
# RANKING PEMAIN
# ==========================
st.header("🏆 Ranking Pemain (Overall)")

df["Skor Ranking"] = (
    df.get("Goal", 0) * 4 +
    df.get("Assist", 0) * 3 +
    df.get("Tembakan", 0) * 1 +
    df.get("Menit Bermain", 0) * 0.002 +
    df.get("Expected Goals", 0) * 3 +
    df.get("Expected Assist", 0) * 3
)

ranking_df = df[["Nama Pemain", "Klub", "Posisi", "Skor Ranking"]].sort_values(
    by="Skor Ranking", ascending=False
).reset_index(drop=True)

ranking_df.insert(0, "Ranking", range(1, len(ranking_df) + 1))

st.subheader("🏅 Ranking 20 Pemain Terbaik")
st.dataframe(ranking_df.head(20))

# ==========================
# TOP 10 PER STATISTIK
# ==========================
st.header("📈 Top 10 Statistik Pemain")

def tampil_top10(col, judul):
    if col in df.columns:
        st.subheader(judul)
        top10 = df[["Nama Pemain", "Klub", col]].sort_values(by=col, ascending=False).head(10)
        top10.insert(0, "Ranking", range(1, len(top10) + 1))
        st.dataframe(top10)

tampil_top10("Goal", "Top 10 Goal")
tampil_top10("Assist", "Top 10 Assist")
tampil_top10("Menit Bermain", "Top 10 Menit Bermain")
tampil_top10("Kartu Merah", "Top 10 Kartu Merah")
tampil_top10("Pelanggaran", "Top 10 Pelanggaran")
tampil_top10("Tembakan", "Top 10 Tembakan")

# ==========================
# HEAD-TO-HEAD
# ==========================
st.header("⚔️ Head-to-Head Pemain")

search1 = st.text_input("Cari Pemain 1:")
search2 = st.text_input("Cari Pemain 2:")

list_pemain = sorted(df["Nama Pemain"].unique())

pilih1 = st.selectbox("Pilih Pemain 1:", [p for p in list_pemain if search1.lower() in p.lower()])
pilih2 = st.selectbox("Pilih Pemain 2:", [p for p in list_pemain if search2.lower() in p.lower()])

if pilih1 and pilih2:
    compare = df[df["Nama Pemain"].isin([pilih1, pilih2])]
    st.subheader(f"Perbandingan Statistik: {pilih1} vs {pilih2}")
    st.dataframe(compare.set_index("Nama Pemain"))

st.success("Semua fitur berhasil dibuat: Default Dataset, Upload Dataset, Ranking, Top 10, Clustering, PCA, dan Head-to-Head.")
