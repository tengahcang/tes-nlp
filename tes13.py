import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Pastikan resource ini sudah diunduh
nltk.download('stopwords')
nltk.download('punkt')

# Baca file CSV
file_path = 'data_lexicon_labeled_numbered.csv'  # Ganti dengan path ke file CSV Anda
data = pd.read_csv(file_path)

# Pastikan kolom ada di CSV
if 'preprocessed_text' not in data.columns or 'label_sentiment_numbered' not in data.columns:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment_numbered' tidak ditemukan dalam CSV!")

# Ambil teks dan label
texts = data['preprocessed_text'].astype(str)  # Pastikan kolom ini berupa string
labels = data['label_sentiment_numbered'].astype(int)  # Pastikan label berupa angka

# Vektorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Model SVM dengan kernel linear
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))
