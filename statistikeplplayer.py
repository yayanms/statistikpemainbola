import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Premier League DSS",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD DATA ====================
@st.cache_resource
def load_and_prepare_data():
    """Load dan prepare data"""
    df = pd.read_csv('fbref_PL_2024-25.csv')
    df_clean = df.dropna(subset=['Gls', 'Ast', 'Min', 'MP', 'xG']).reset_index(drop=True)
    
    feature_columns = ['Gls', 'Ast', 'Min', 'MP', 'xG']
    X = df_clean[feature_columns].copy()
    
    # Feature Engineering
    df_clean['G+A'] = df_clean['Gls'] + df_clean['Ast']
    df_clean['Minutes_90'] = df_clean['Min'] / 90
    df_clean['G+A_per_90'] = df_clean['G+A'] / df_clean['Minutes_90'].replace(0, 1)
    
    def classify_performance(g_a_per_90):
        if g_a_per_90 >= 0.3:
            return 'Pemain Berpotensi'
        else:
            return 'Pemain Kurang Berpotensi'
    
    df_clean['Performance_Class'] = df_clean['G+A_per_90'].apply(classify_performance)
    
    # Normalize & Train KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df_clean['Performance_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    return df_clean, feature_columns, scaler, knn_model, X_test, y_test

df_clean, feature_columns, scaler, knn_model, X_test, y_test = load_and_prepare_data()

# ==================== HELPER FUNCTIONS ====================
def predict_player(player_name):
    """Prediksi performa pemain berdasarkan nama"""
    if player_name not in df_clean['Player'].values:
        return None
    
    idx = df_clean[df_clean['Player'] == player_name].index[0]
    player_row = df_clean.loc[idx]
    
    player_features = np.array(player_row[feature_columns]).reshape(1, -1)
    player_features_scaled = scaler.transform(player_features)
    
    prediction = knn_model.predict(player_features_scaled)[0]
    probabilities = knn_model.predict_proba(player_features_scaled)[0]
    confidence = max(probabilities) * 100
    
    distances, indices = knn_model.kneighbors(player_features_scaled)
    neighbors = [df_clean.iloc[idx]['Player'] for idx in indices[0]]
    
    return {
        'nama': player_name,
        'klub': player_row['Squad'],
        'posisi': player_row['Pos'],
        'prediksi': prediction,
        'gol': int(player_row['Gls']),
        'assist': int(player_row['Ast']),
        'menit': int(player_row['Min']),
        'g_a_per_90': round(player_row['G+A_per_90'], 3),
        'confidence': round(confidence, 2),
        'similar_players': neighbors[:5]
    }

def get_color_class(performance_class):
    """Get emoji untuk performance class"""
    colors = {
        'Pemain Berpotensi': '✅',
        'Pemain Kurang Berpotensi': '❌'
    }
    return colors.get(performance_class, '❓')

# ==================== HEADER ====================
st.title('⚽ Premier League Player Performance DSS')
st.markdown('**Sistem Pendukung Keputusan - Klasifikasi Performa Pemain dengan K-Nearest Neighbors**')

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header('📊 Menu Navigasi')
    menu = st.radio(
        'Pilih Halaman:',
        ['📈 Dashboard', '🏆 Pilih Kategori Pemain', '🔍 Analisis Lengkap Pemain', '📋 Evaluasi Model']
    )

# ==================== PAGE 1: DASHBOARD ====================
if menu == '📈 Dashboard':
    st.header('📈 Dashboard Umum')
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pemain", len(df_clean), "pemain")
    with col2:
        st.metric("Rata-rata Gol", f"{df_clean['Gls'].mean():.2f}", "per pemain")
    with col3:
        st.metric("Rata-rata Assist", f"{df_clean['Ast'].mean():.2f}", "per pemain")
    with col4:
        st.metric("Rata-rata G+A/90", f"{df_clean['G+A_per_90'].mean():.3f}", "per 90 menit")
    
    st.divider()
    
    # Top Scorers
    st.subheader('⭐ Top 15 Scorers')
    top_scorers = df_clean.nlargest(15, 'Gls')[['Player', 'Squad', 'Pos', 'Gls', 'Ast', 'Performance_Class']]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        for idx, row in top_scorers.iterrows():
            emoji = get_color_class(row['Performance_Class'])
            st.write(f"{emoji} **{row['Player']}** | {row['Squad']} | {row['Pos']} | **⚽{int(row['Gls'])} 🅰️{int(row['Ast'])}**")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.barh(top_scorers['Player'], top_scorers['Gls'], color='#FF6B6B')
        ax.set_xlabel('Gol')
        ax.set_title('Top 15 Scorers')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.divider()
    
    # Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📊 Distribusi Kelas Performa')
        class_dist = df_clean['Performance_Class'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#10b981', '#ef4444']
        ax.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Distribusi Kelas Performa')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader('🎯 G+A per 90 Distribution')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df_clean['G+A_per_90'].dropna(), bins=30, color='#45B7D1', edgecolor='black', alpha=0.7)
        ax.set_xlabel('G+A per 90')
        ax.set_ylabel('Jumlah Pemain')
        ax.set_title('Distribusi G+A per 90 Menit')
        ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Threshold: 0.3')
        ax.legend()
        st.pyplot(fig, use_container_width=True)

# ==================== PAGE 2: PILIH KATEGORI PEMAIN ====================
elif menu == '🏆 Pilih Kategori Pemain':
    st.header('🏆 Pilih Kategori Pemain')
    
    # Initialize session state
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = None
    
    st.subheader('📂 Pilih Kategori:')
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('✅ Pemain Berpotensi', use_container_width=True):
            st.session_state.selected_category = 'Pemain Berpotensi'
            st.rerun()
    
    with col2:
        if st.button('❌ Pemain Kurang Berpotensi', use_container_width=True):
            st.session_state.selected_category = 'Pemain Kurang Berpotensi'
            st.rerun()
    
    st.divider()
    
    # Pilih ranking
    st.subheader('🎯 Pilih Fitur Ranking:')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('⚽ Goal Terbanyak', use_container_width=True):
            st.session_state.selected_metric = 'Gls'
            st.rerun()
    
    with col2:
        if st.button('🅰️ Assist Terbanyak', use_container_width=True):
            st.session_state.selected_metric = 'Ast'
            st.rerun()
    
    with col3:
        if st.button('🎯 G+A Terbanyak', use_container_width=True):
            st.session_state.selected_metric = 'G+A'
            st.rerun()
    
    st.divider()
    
    # Validasi input
    if 'selected_category' not in st.session_state or st.session_state.selected_category is None:
        st.warning('⚠️ Silakan pilih **Kategori Pemain** terlebih dahulu')
    elif 'selected_metric' not in st.session_state or st.session_state.selected_metric is None:
        st.warning('⚠️ Silakan pilih **Fitur Ranking** terlebih dahulu')
    else:
        category = st.session_state.selected_category
        metric = st.session_state.selected_metric
        
        # Filter data
        filtered_data = df_clean[df_clean['Performance_Class'] == category].copy()
        
        if len(filtered_data) > 0:
            # Sort berdasarkan metric
            filtered_data = filtered_data.sort_values(metric, ascending=False).head(15)
            
            st.subheader(f'📊 Top 15 - {category} (Ranking: {metric})')
            st.info(f'Total pemain di kategori ini: {len(df_clean[df_clean["Performance_Class"] == category])}')
            
            # Tabel
            display_cols = ['Player', 'Squad', 'Pos', 'Gls', 'Ast', 'G+A', 'Min', 'G+A_per_90']
            st.dataframe(filtered_data[display_cols], use_container_width=True)
            
            # Grafik
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(filtered_data['Player'], filtered_data[metric], color='#4ECDC4')
                ax.set_xlabel(metric)
                ax.set_title(f'Top 15 - {category} (by {metric})')
                ax.invert_yaxis()
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.subheader('📈 Statistik Kategori')
                st.write(f"**Kategori:** {category}")
                st.write(f"**Total Pemain:** {len(df_clean[df_clean['Performance_Class'] == category])}")
                st.write(f"**Rata-rata Gol:** {filtered_data['Gls'].mean():.2f}")
                st.write(f"**Rata-rata Assist:** {filtered_data['Ast'].mean():.2f}")
                st.write(f"**Rata-rata G+A per 90:** {filtered_data['G+A_per_90'].mean():.3f}")
            
            st.divider()
            
            # Analisis detail
            st.subheader('🔍 Analisis Detail Pemain')
            col_select, col_btn = st.columns([3, 1])
            
            with col_select:
                selected_player = st.selectbox(
                    'Pilih pemain untuk melihat detail:',
                    filtered_data['Player'].tolist(),
                    key='player_select'
                )
            
            with col_btn:
                st.write('')
                analyze_btn = st.button('📊 Analisis', use_container_width=True, key='analyze_btn')
            
            # Info pemain langsung muncul
            if selected_player:
                result = predict_player(selected_player)
                
                if result:
                    # Header info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Nama", result['nama'])
                    with col2:
                        st.metric("Klub", result['klub'])
                    with col3:
                        st.metric("Posisi", result['posisi'])
                    with col4:
                        st.metric("Prediksi", result['prediksi'])
                    
                    # Detail hanya muncul saat klik Analisis
                    if analyze_btn:
                        st.divider()
                        
                        emoji = get_color_class(result['prediksi'])
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader('📊 Klasifikasi Performa')
                            st.markdown(f"### {emoji} {result['prediksi']}")
                            st.write('**Confidence Level:**')
                            st.progress(result['confidence'] / 100)
                            st.write(f"**{result['confidence']}%**")
                        
                        with col2:
                            st.subheader('📈 Statistik')
                            col_stat1, col_stat2 = st.columns(2)
                            with col_stat1:
                                st.metric("Gol", result['gol'])
                                st.metric("Assist", result['assist'])
                            with col_stat2:
                                st.metric("Menit", result['menit'])
                                st.metric("G+A per 90", result['g_a_per_90'])
                        
                        st.divider()
                        
                        # Similar Players
                        st.subheader('🤝 Pemain Serupa (K-Nearest Neighbors)')
                        cols = st.columns(len(result['similar_players']))
                        for idx, col in enumerate(cols):
                            with col:
                                similar_player = result['similar_players'][idx]
                                similar_data = df_clean[df_clean['Player'] == similar_player].iloc[0]
                                st.markdown(f"""
                                **{similar_player}**
                                
                                {similar_data['Squad']} | {similar_data['Pos']}
                                
                                ⚽ {int(similar_data['Gls'])} | 🅰️ {int(similar_data['Ast'])}
                                
                                {similar_data['G+A_per_90']:.3f} G+A/90
                                """)
        else:
            st.error(f"❌ Tidak ada pemain di kategori '{category}'")

# ==================== PAGE 3: ANALISIS LENGKAP ====================
elif menu == '🔍 Analisis Lengkap Pemain':
    st.header('🔍 Analisis Lengkap Pemain')
    st.markdown('**Cari dan analisis pemain manapun**')
    
    st.subheader('🔎 Cari Pemain:')
    selected_player = st.selectbox(
        'Pilih atau ketik nama pemain:',
        sorted(df_clean['Player'].tolist()),
        key='full_analysis_player'
    )
    
    if selected_player:
        result = predict_player(selected_player)
        
        if result:
            st.subheader('📋 Informasi Pemain')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nama", result['nama'])
            with col2:
                st.metric("Klub", result['klub'])
            with col3:
                st.metric("Posisi", result['posisi'])
            with col4:
                st.metric("Prediksi", result['prediksi'])
            
            st.divider()
            
            # Klasifikasi
            st.subheader('✅ Klasifikasi Performa')
            emoji = get_color_class(result['prediksi'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {emoji} {result['prediksi']}")
            
            with col2:
                st.write('**Confidence:**')
                st.progress(result['confidence'] / 100)
                st.write(f"{result['confidence']}%")
            
            with col3:
                st.metric("G+A per 90", result['g_a_per_90'])
            
            st.divider()
            
            # Statistik
            st.subheader('📊 Statistik Detail')
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Gol", result['gol'])
                st.metric("Assist", result['assist'])
            
            with col2:
                st.metric("Menit Bermain", result['menit'])
                st.metric("Total G+A", result['gol'] + result['assist'])
            
            st.divider()
            
            # Similar Players
            st.subheader('🤝 Pemain Serupa')
            cols = st.columns(len(result['similar_players']))
            for idx, col in enumerate(cols):
                with col:
                    similar_player = result['similar_players'][idx]
                    similar_data = df_clean[df_clean['Player'] == similar_player].iloc[0]
                    st.markdown(f"""
                    **{similar_player}**
                    
                    {similar_data['Squad']} | {similar_data['Pos']}
                    
                    ⚽ {int(similar_data['Gls'])} | 🅰️ {int(similar_data['Ast'])}
                    """)

# ==================== PAGE 4: EVALUASI MODEL ====================
elif menu == '📋 Evaluasi Model':
    st.header('📋 Evaluasi Model KNN')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('🎯 Akurasi Model')
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy*100:.2f}%", f"{len(y_test)} test samples")
        
        st.subheader('📊 Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']], use_container_width=True)
    
    with col2:
        st.subheader('🔥 Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        classes = sorted(df_clean['Performance_Class'].unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>⚽ Premier League Player Performance Decision Support System (DSS)</p>
    <p>K-Nearest Neighbors (KNN) Binary Classification</p>
    <p>2 Class: Pemain Berpotensi (G+A/90 ≥ 0.3) | Pemain Kurang Berpotensi (G+A/90 < 0.3)</p>
</div>
""", unsafe_allow_html=True)