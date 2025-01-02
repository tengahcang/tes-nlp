import pandas as pd

# 1. Membaca file CSV
file_path = "undersampled_data.csv"  # Ganti dengan file Anda
data = pd.read_csv(file_path)

# 2. Mapping label sentimen ke nilai numerik
label_mapping = {
    'positif': 2,
    'netral': 1,
    'negatif': 0
}
data['label_sentiment'] = data['label_sentiment'].replace(label_mapping)


# 3. Menyimpan hasil ke file baru
output_file = "numerical_labels_data.csv"
data.to_csv(output_file, index=False)

print("Label sentimen berhasil diubah!")
print(data.head())
