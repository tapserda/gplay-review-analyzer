# pip install google-play-scraper wordcloud nltk seaborn scikit-learn

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from google_play_scraper import Sort, reviews
from wordcloud import WordCloud
from datetime import datetime, timedelta

# Penanganan inisialisasi NLTK untuk menghindari error circular import
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
except:
    import nltk.data
    from nltk.corpus import stopwords
    nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# KONFIGURASI APLIKASI (BISA DIGANTI)
# ==========================================
# Contoh: 'com.mobile.legends', 'com.shopee.id', 'com.whatsapp'
app_id = 'com.mobile.legends' 
target_data = 20000  # Maksimal ulasan yang diambil
rentang_hari = 30     # Batas waktu ke belakang (hari)
# ==========================================

# --- 1. SETUP PEMBERSIHAN TEKS (NLP) ---
stop_factory = stopwords.words('indonesian')
# Menambahkan kata-kata yang tidak memiliki nilai informatif dalam analisis
tambahan_stopword = ['aj', 'aja', 'gua', 'gw', 'gak', 'gk', 'yg', 'ga', 'kalo', 'udah', 'dah', 'bgt', 'game', 'nya', 'dan', 'ke', 'di', 'ini', 'itu', 'saya', 'aku', 'aplikasi']
list_stopwords = set(stop_factory + tambahan_stopword)

def clean_text(text):
    """Fungsi untuk membersihkan teks ulasan dari noise"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Hapus simbol & angka
    text = ' '.join([word for word in text.split() if word not in list_stopwords])
    return text

# --- 2. DATA COLLECTION (SCRAPING) ---
limit_tanggal = datetime.now() - timedelta(days=rentang_hari)
all_reviews = []
token = None

print(f"--- Memulai Scraping Aplikasi: {app_id} ---")
print(f"Target: Hingga {rentang_hari} hari ke belakang...")

while len(all_reviews) < target_data:
    rvs, token = reviews(
        app_id, lang='id', country='id',
        sort=Sort.NEWEST, count=200, continuation_token=token
    )
    all_reviews.extend(rvs)
    last_date = rvs[-1]['at']
    print(f"Terkumpul {len(all_reviews)} data... (Hingga ulasan tanggal: {last_date.strftime('%Y-%m-%d')})")
    
    if last_date < limit_tanggal or not token:
        break

# Simpan ke DataFrame dan filter sesuai rentang waktu
df_full = pd.DataFrame(all_reviews)
df_full = df_full[df_full['at'] >= limit_tanggal].copy()
df_full['content_clean'] = df_full['content'].apply(clean_text)
df_full['label'] = df_full['score'].apply(lambda x: 1 if x > 3 else 0)

# --- 3. MACHINE LEARNING MODELING ---
# Mengubah teks menjadi fitur angka (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_full['content_clean'])
y = df_full['label']

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menggunakan Logistic Regression karena efisien untuk teks
model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nModel AI Siap! Akurasi Prediksi: {acc*100:.2f}%")

# --- 4. ANALISIS INTERAKTIF ---
print("\n" + "="*40)
print(f"MODE ANALISIS UNTUK {app_id.upper()}")
print("1. Analisis Seluruh Data (30 Hari)")
print("2. Analisis Tanggal Spesifik")
menu = input("Pilih menu (1/2): ")
print("="*40)

if menu == '1':
    df_target = df_full.copy()
    judul_grafik = f"Analisis 30 Hari - {app_id}"
elif menu == '2':
    tgl_user = input("Masukkan tanggal (Format YYYY-MM-DD): ")
    df_full['at_str'] = df_full['at'].dt.strftime('%Y-%m-%d')
    df_target = df_full[df_full['at_str'] == tgl_user].copy()
    judul_grafik = f"Analisis Tanggal {tgl_user} - {app_id}"
else:
    print("Pilihan salah, default ke Analisis 30 Hari.")
    df_target = df_full.copy()
    judul_grafik = f"Analisis 30 Hari - {app_id}"

# --- 5. VISUALISASI ---
if not df_target.empty:
    X_target = vectorizer.transform(df_target['content_clean'])
    df_target['prediksi'] = model.predict(X_target)

    # A. PIE CHART (SENTIMEN)
    plt.figure(figsize=(7, 7))
    counts = df_target['prediksi'].value_counts()
    plt.pie(counts, labels=['Negatif' if i == 0 else 'Positif' for i in counts.index], 
            autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=140)
    plt.title(f'Proporsi Sentimen: {judul_grafik}')
    plt.show()

    # B. BIGRAM (KONTEKS POSITIF VS NEGATIF)
    def plot_bg(data, title, cmap, ax_obj):
        if len(data) > 1:
            cv = CountVectorizer(ngram_range=(2, 2), max_features=10)
            try:
                bg_matrix = cv.fit_transform(data)
                res = pd.DataFrame({'Bigram': cv.get_feature_names_out(), 'Jumlah': bg_matrix.toarray().sum(axis=0)})
                res = res.sort_values(by='Jumlah', ascending=False)
                sns.barplot(ax=ax_obj, x='Jumlah', y='Bigram', data=res, hue='Bigram', palette=cmap, legend=False)
                ax_obj.set_title(title)
            except: ax_obj.set_title(f"{title} (Gagal mengekstrak kata)")
        else: ax_obj.set_title(f"{title} (Data tidak cukup)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    plot_bg(df_target[df_target['prediksi'] == 0]['content_clean'], 'Masalah Terbanyak (Negatif)', 'Reds_r', ax1)
    plot_bg(df_target[df_target['prediksi'] == 1]['content_clean'], 'Apresiasi Terbanyak (Positif)', 'Blues_r', ax2)
    plt.tight_layout()
    plt.show()

    # C. WORD CLOUD
    wc = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(' '.join(df_target['content_clean']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis('off')
    plt.title(f'Word Cloud Dominan: {judul_grafik}')
    plt.show()
else:
    print("Tidak ada data untuk dianalisis.")
