import pandas as pd
import re
import json
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def simple_tokenize(text):
    return text.split()

def normalize_slang(text, slang_dict):
    words = simple_tokenize(text)
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def preprocess_text(text, slang_dict, stemmer):
    text = text.lower()

    text = normalize_slang(text, slang_dict)

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'@\w+\s*', '', text) 
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+|http\w+', '', text)
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('\(.*?\)', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('[‘’“”…♪♪]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('b ', ' ', text)
    text = re.sub('rt ', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    tokens = simple_tokenize(text)

    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

def load_slang_dict(file_txt):
    with open(file_txt, 'r', encoding='utf-8') as file:
        slang_dict = json.load(file)

    return slang_dict

slang_path = 'slank_word_dictionary.txt'
slang_dict = load_slang_dict(slang_path)

path = 'full_text_output.csv'  
df = pd.read_csv(path)

df['preprocessed_text'] = df['full_text'].apply(lambda x: preprocess_text(x, slang_dict, stemmer))

df[['preprocessed_text']].to_csv('preprocessed_text_output.csv', index=False)

print("Preprocessing selesai, hasil kolom 'preprocessed_text' disimpan ke 'preprocessed_text_output.csv'")
