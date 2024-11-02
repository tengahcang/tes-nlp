import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Unduh lexicon VADER dan punkt untuk tokenisasi
nltk.download('punkt')
nltk.download('vader_lexicon')

# Inisialisasi Stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi SentimentIntensityAnalyzer
senti_indo = SentimentIntensityAnalyzer()

# Contoh teks yang ingin diproses
preprocessed_text = "Saya merasa sangat senang hari ini tetapi juga sedikit negatif tentang beberapa hal."

# Stemming pada teks
stemmed_text = stemmer.stem(preprocessed_text)
print("Teks setelah stemming:", stemmed_text)

# Tokenisasi teks yang sudah distem
tokens = word_tokenize(stemmed_text)
for token in tokens:
    print("Kata:", token)
    score = senti_indo.polarity_scores(token)
    
    if token in senti_indo.lexicon:
        print("Nilai lexicon:", senti_indo.lexicon[token])
    else:
        print("Nilai lexicon: 0")
    
    print("Score negatif:", score['neg'])
    print("Score netral:", score['neu'])
    print("Score positif:", score['pos'])
    print("Score compound:", score['compound'])
    print()  # Tambahkan baris kosong untuk pemisah antar kata

# Skor untuk seluruh teks
overall_score = senti_indo.polarity_scores(stemmed_text)
print("Skor keseluruhan untuk teks:", overall_score)
