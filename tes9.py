import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Unduh lexicon VADER
nltk.download('vader_lexicon')

# Path ke file lexicon bahasa Indonesia
path = 'sentiwords_id.txt'  # Pastikan file ini sudah ada
df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])

# Buat dictionary dari lexicon bahasa Indonesia
senti_dict = {row['word']: float(row['value']) for _, row in df_senti.iterrows()}

# Inisialisasi Stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi SentimentIntensityAnalyzer dan update lexicon-nya
senti_indo = SentimentIntensityAnalyzer()
senti_indo.lexicon.update(senti_dict)

# Contoh teks yang ingin diproses
preprocessed_text = "Saya merasa sangat senang hari ini tetapi juga sedikit negatif tentang beberapa hal."

# Stemming pada teks
stemmed_text = stemmer.stem(preprocessed_text)
print("Teks setelah stemming:", stemmed_text)

# Tokenisasi menggunakan regex
def tokenize(text):
    return re.findall(r'\w+', text)

# Tokenisasi teks yang sudah distem
tokens = tokenize(stemmed_text)

# Analisis sentimen per kata setelah stemming
print("\nAnalisis Sentimen per Kata:")
for token in tokens:
    score = senti_indo.polarity_scores(token)
    print(f"Kata: {token}")
    if token in senti_indo.lexicon:
        print(f"Nilai lexicon: {senti_indo.lexicon[token]}")
    else:
        print("Nilai lexicon: 0")
    print("Score negatif:", score['neg'])
    print("Score netral:", score['neu'])
    print("Score positif:", score['pos'])
    print("Score compound:", score['compound'])
    print()  # Tambahkan baris kosong untuk pemisah antar kata

# Analisis sentimen untuk seluruh teks
overall_score = senti_indo.polarity_scores(stemmed_text)
print("\nSkor keseluruhan untuk teks:", overall_score)


if overall_score['compound'] >= 0.05:
    print("Sentimen Keseluruhan: Positive")
elif overall_score['compound'] <= -0.05:
    print("Sentimen Keseluruhan: Negative")
else:
    print("Sentimen Keseluruhan: Neutral")