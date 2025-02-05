import pandas as pd

path = 'data_X_kurikulum merdeka.csv'
df = pd.read_csv(path)

print(df['full_text'])

df['full_text'].to_csv('full_text_output.csv', index=False)

print("Kolom 'full_text' berhasil disimpan ke 'full_text_output.csv'")