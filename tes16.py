import pandas as pd
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Pastikan resource ini sudah diunduh
nltk.download('stopwords')
nltk.download('punkt')

# Baca file CSV
file_path = 'numerical_labels_data.csv'  # Ganti dengan path ke file CSV Anda
data = pd.read_csv(file_path)

# Pastikan kolom ada di CSV
if 'preprocessed_text' not in data.columns or 'label_sentiment' not in data.columns:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment' tidak ditemukan dalam CSV!")

# Ambil teks dan label
texts = data['preprocessed_text'].astype(str)  # Pastikan kolom ini berupa string
labels = data['label_sentiment'].astype(int)  # Pastikan label berupa angka

# Vektorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
# print(vectorizer)
X = vectorizer.fit_transform(texts)
# print(X)
# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=42)


# 1. SVM (Support Vector Machine)
print("Support Vector Machine (SVM)")
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))


# 3. Random Forest Classifier
print("\nRandom Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

df_test = data.iloc[y_test.index].copy()  # Menggunakan index yang sama dengan data uji

df_test['predicted_label_svm'] = y_pred_svm
df_test['predicted_label_rf'] = y_pred_rf


# Menampilkan DataFrame dengan kolom baru
print(df_test.head())

# Menyimpan DataFrame ke file CSV jika diperlukan
output_file_path = 'hasil_prediksi_berbagai_model.csv'  # Tentukan path untuk file output
df_test.to_csv(output_file_path, index=False)
print(f"Hasil prediksi telah disimpan ke {output_file_path}")

# Path untuk menyimpan model
svm_model_path = "svm_model.pkl"

# Simpan model SVM ke file pickle
with open(svm_model_path, 'wb') as svm_model_file:
    pickle.dump(svm_clf, svm_model_file)

print(f"Model SVM telah disimpan ke {svm_model_path}")


# Path untuk menyimpan model
rf_model_path = "rf_model.pkl"

# Simpan model SVM ke file pickle
with open(rf_model_path, 'wb') as rf_model_file:
    pickle.dump(rf_clf, rf_model_file)

print(f"Model Random Forest Classifier telah disimpan ke {rf_model_path}")

# Path untuk menyimpan model
vectorizer_model_path = "vectorizer_model.pkl"

# Simpan model SVM ke file pickle
with open(vectorizer_model_path, 'wb') as vectorizer_model_file:
    pickle.dump(vectorizer, vectorizer_model_file)

print(f"Model vectorizer telah disimpan ke {vectorizer_model_path}")