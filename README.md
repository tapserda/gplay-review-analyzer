# 📊 Universal Google Play Review Analyzer

Alat ini memungkinkan pengguna untuk melakukan analisis sentimen mendalam pada aplikasi Android apa pun. Dengan mengekstrak hingga 20.000 ulasan terbaru, alat ini memberikan gambaran visual tentang apa yang disukai dan dikeluhkan oleh pengguna secara akurat menggunakan Machine Learning.

## 🛠️ Cara Kerja
1. **Scraping**: Mengambil data ulasan terbaru menggunakan `google-play-scraper`.
2. **NLP & Cleaning**: Membersihkan teks ulasan dari noise (emoji, stopwords, tanda baca).
3. **Sentiment Analysis**: Menggunakan *Logistic Regression* (Trained on TF-IDF features) untuk memprediksi sentimen ulasan.
4. **Context Extraction**: Menggunakan Bigram untuk mengetahui alasan spesifik di balik sentimen positif/negatif.

## ⚙️ Penggunaan
Ganti nilai variabel `app_id` di baris ke-35 untuk menganalisis aplikasi lain.
