import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Path ke file model dan vectorizer
svm_model_path = "svm_model.pkl"  # Path file model SVM
vectorizer_path = "vectorizer_model.pkl"  # Path file vectorizer

# Memuat model SVM dari file pickle
with open(svm_model_path, 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)
print("Model SVM telah berhasil dimuat kembali!")

# Memuat TfidfVectorizer dari file pickle
with open(vectorizer_path, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)
print("Vectorizer telah berhasil dimuat kembali!")

# Contoh prediksi menggunakan model dan vectorizer yang dimuat
sample_texts = ["kurikulum merdeka itu letak merdeka mana ya"]
sample_features = loaded_vectorizer.transform(sample_texts)  # Vektorisasi teks baru
predictions = loaded_svm_model.predict(sample_features)

print("Hasil prediksi:", predictions)  # Output: Label sentiment untuk teks sample
