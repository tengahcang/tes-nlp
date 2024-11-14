import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Pastikan resource ini sudah diunduh
nltk.download('stopwords')
nltk.download('punkt')

# Contoh data dengan 3 label: 2 = Positif, 1 = Netral, 0 = Negatif
texts = [
    "saya senang belajar",       # Positif
    "ini adalah bencana",        # Negatif
    "saya sangat senang",        # Positif
    "saya tidak suka ini",       # Negatif
    "saya merasa biasa saja",    # Netral
    "ini adalah hari yang baik", # Positif
    "saya tidak punya pendapat", # Netral
    "ini cukup membosankan"      # Negatif
]
labels = [2, 0, 2, 0, 1, 2, 1, 0]  # Label sesuai dengan teks

# Langkah praproses: vektorisasi dengan TF-IDF
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
