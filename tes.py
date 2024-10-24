import pandas as pd
import nltk
# # Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re


path = 'data_X_kurikulum merdeka.csv'
df = pd.read_csv(path)

print(df['full_text'])

df['full_text'].to_csv('full_text_output.csv', index=False)

print("Kolom 'full_text' berhasil disimpan ke 'full_text_output.csv'")