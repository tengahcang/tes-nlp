import pandas as pd

# Baca kedua file CSV
csv1 = pd.read_csv('data_X_kurikulum merdeka.csv')  # Ganti dengan nama file CSV pertama
csv2 = pd.read_csv('Kurikulum_Merdeka_28-12-2024_10-30-36.csv')  # Ganti dengan nama file CSV kedua

# Gabungkan kedua data
combined = pd.concat([csv1, csv2])

# Hapus duplikat berdasarkan semua kolom
unique_data = combined.drop_duplicates(subset=['full_text'])

# Simpan hasilnya ke file baru
unique_data.to_csv('combined_unique.csv', index=False)

print("Data berhasil digabungkan dan disimpan di 'combined_unique.csv'")
