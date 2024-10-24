import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re
import gdown
import json

# Mendownload resource yang dibutuhkan
nltk.download('stopwords')
nltk.download('punkt')  # Pastikan 'punkt' diunduh

# Fungsi untuk mendownload dan memuat kamus slang dari Google Drive
def load_slang_dict(drive_url):
    # Extract the file ID from the Google Drive URL
    file_id = drive_url.split('/')[-2]

    # Construct the download URL
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Download the file using gdown (it will be saved locally as 'slang_dict.txt')
    gdown.download(download_url, 'slang_dict.txt', quiet=False)

    # Now open the downloaded file and load it as a dictionary
    with open('slang_dict.txt', 'r', encoding='utf-8') as file:
        slang_dict = json.load(file)

    return slang_dict

# Menambah slang dictionary baru
new_slang_dict = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "kagak": "tidak",
    "enggak": "tidak",
}

# Fungsi untuk normalisasi slang dalam teks
def normalize_slang(text, slang_dict):
    words = word_tokenize(text)  # Tokenize words
    normalized_words = [slang_dict.get(word, word) for word in words]  # Replace slang if found in dictionary
    return ' '.join(normalized_words)

# Fungsi praproses teks termasuk normalisasi slang, tokenisasi, penghapusan stopwords, dan stemming
def preprocess_text(text, slang_dict, stemmer):
    # Lowercasing
    text = text.lower()

    # Normalize slang and abbreviations
    text = normalize_slang(text, slang_dict)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    custom_stopwords = set(stopwords.words('indonesian')).union(set(stopwords.words('english')))
    new_stopwords = ['habis']
    custom_stopwords.update(new_stopwords)
    exclude_words = ['dikerjai', 'tidak']  # Words you don't want to exclude
    custom_stopwords = custom_stopwords.difference(exclude_words)
    tokens = [word for word in tokens if word not in custom_stopwords]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a single string
    return ' '.join(tokens)

# Membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load data CSV
path = 'data_X_kurikulum merdeka.csv'
df = pd.read_csv(path)

# Cetak kolom 'full_text' sebelum praproses
print("Sebelum Praproses:")
print(df['full_text'])

# Simpan kolom 'full_text' ke file baru
df['full_text'].to_csv('full_text_output.csv', index=False)
print("Kolom 'full_text' berhasil disimpan ke 'full_text_output.csv'")

# Asumsikan kita sudah memuat kamus slang dari URL Google Drive
# Untuk contoh ini, kita langsung pakai new_slang_dict
slang_dict = new_slang_dict

# Proses setiap teks di kolom 'full_text'
df['preprocessed_text'] = df['full_text'].apply(lambda x: preprocess_text(str(x), slang_dict, stemmer))

# Cetak hasil setelah praproses
print("Setelah Praproses:")
print(df['preprocessed_text'])

# Simpan hasil praproses ke file CSV baru
df[['preprocessed_text']].to_csv('preprocessed_text_output.csv', index=False)
print("Kolom 'preprocessed_text' berhasil disimpan ke 'preprocessed_text_output.csv'")
