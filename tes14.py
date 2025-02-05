import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# 1. Membaca file CSV
file_path = "data_lexicon_labeled.csv"  # Sesuaikan dengan lokasi file Anda
data = pd.read_csv(file_path)

# 2. Memisahkan fitur dan label
# Misalkan kolom 'text' berisi data teks dan kolom 'label' berisi sentimen
X = data['preprocessed_text']
y = data['label_sentiment']

# Konversi ke DataFrame karena RandomUnderSampler membutuhkan array
X = X.values.reshape(-1, 1)  # Mengubah ke bentuk (n_samples, n_features)
y = y.values

# 3. Menerapkan undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 4. Mengubah kembali hasil resampling ke DataFrame
undersampled_data = pd.DataFrame({
    'preprocessed_text': X_resampled.flatten(),
    'label_sentiment': y_resampled
})


undersampled_data.to_csv("undersampled_data.csv", index=False)

print("Data sebelum undersampling:")
print(data['label_sentiment'].value_counts())
print("\nData setelah undersampling:")
print(undersampled_data['label_sentiment'].value_counts())
