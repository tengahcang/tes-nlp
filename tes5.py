import pandas as pd
import re
import json
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Membuat stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Tokenisasi sederhana menggunakan spasi
def simple_tokenize(text):
    return text.split()

# Fungsi untuk normalisasi slang
def normalize_slang(text, slang_dict):
    words = simple_tokenize(text)  # Tokenisasi manual
    normalized_words = [slang_dict.get(word, word) for word in words]  # Ganti slang dengan kata yang benar
    return ' '.join(normalized_words)

# Fungsi untuk preprocessing teks
def preprocess_text(text, slang_dict, stemmer):
    # Lowercasing
    text = text.lower()

    # Normalisasi slang dan singkatan
    text = normalize_slang(text, slang_dict)

    # Hilangkan tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'@\w+\s*', '', text)  # Hapus mention
    text = re.sub(r'https?://\S+', '', text)  # Hapus link
    text = re.sub(r'https?://\S+|www\.\S+|http\w+', '', text)  # Hapus link termasuk variasi tanpa spasi
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('\(.*?\)', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # Hapus tanda baca
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('[‘’“”…♪♪]', '', text)  # Hapus karakter spesial
    text = re.sub('\n', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('b ', ' ', text)
    text = re.sub('rt ', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Tokenisasi manual
    tokens = simple_tokenize(text)

    # Stemming menggunakan Sastrawi
    tokens = [stemmer.stem(word) for word in tokens]

    # Gabungkan token kembali menjadi kalimat
    return ' '.join(tokens)

# Tambahkan dictionary slang yang diinginkan
# slang_dict = {
#     "gak": "tidak",
#     "ga": "tidak",
#     "nggak": "tidak",
#     "g": "tidak",
#     "kagak": "tidak",
#     "enggak": "tidak",
# }

def load_slang_dict(file_txt):
    # Now open the downloaded file and load it as a dictionary
    with open(file_txt, 'r', encoding='utf-8') as file:
        slang_dict = json.load(file)

    return slang_dict

slang_path = 'slang_abbrevations_words.txt'
slang_dict = load_slang_dict(slang_path)

# Load data dari 'full_text_output.csv'
path = 'full_text_output.csv'  # Ini file yang sudah Anda buat sebelumnya
df = pd.read_csv(path)

# Terapkan preprocessing pada kolom 'full_text'
df['preprocessed_text'] = df['full_text'].apply(lambda x: preprocess_text(x, slang_dict, stemmer))

# Simpan hanya kolom 'preprocessed_text' ke file CSV baru
df[['preprocessed_text']].to_csv('preprocessed_text_output.csv', index=False)

print("Preprocessing selesai, hasil kolom 'preprocessed_text' disimpan ke 'preprocessed_text_output.csv'")
