import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Buat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def simple_tokenize(text):
    # Tokenisasi sederhana dengan spasi
    return text.split()

def preprocess_text(text, slang_dict, stemmer):
    # Lowercasing
    text = text.lower()

    # Normalize slang and abbreviations
    text = normalize_slang(text, slang_dict)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text using simple tokenization
    tokens = simple_tokenize(text)

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a single string
    return ' '.join(tokens)

def normalize_slang(text, slang_dict):
    # Tokenisasi dan normalisasi slang
    words = simple_tokenize(text)
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Contoh teks
text = "Ini contoh teks bahasa Indonesia yang perlu diproses."

# Dictionary slang yang sudah diupdate
slang_dict = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "kagak": "tidak",
    "enggak": "tidak",
}

# Preprocessing
preprocessed_text = preprocess_text(text, slang_dict, stemmer)
print(preprocessed_text)
