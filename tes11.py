import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

nltk.download('vader_lexicon')


path = 'sentiwords_id.txt' 
df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])

senti_dict = {row['word']: float(row['value']) for _, row in df_senti.iterrows()}

factory = StemmerFactory()
stemmer = factory.create_stemmer()

senti_indo = SentimentIntensityAnalyzer()
senti_indo.lexicon.update(senti_dict)

df = pd.read_csv('preprocessed_text_output.csv')

label_lexicon = []

def tokenize(text):
    return re.findall(r'\w+', text)

for index, row in df.iterrows():
    stemmed_text = stemmer.stem(row['preprocessed_text'])
    score = senti_indo.polarity_scores(stemmed_text)
    if score['compound'] >= 0.05:
        label_lexicon.append(2)  # positif
    elif score['compound'] <= -0.05:
        label_lexicon.append(0)  # negatif
    else:
        label_lexicon.append(1)  # netral
df['label_sentiment_numbered'] = label_lexicon

df.to_csv('data_lexicon_labeled_numbered.csv', index=False)

print("Proses selesai! Hasil sentimen telah disimpan di 'data_lexicon_labeled_numbered.csv'")

print(df['label_sentiment_numbered'].value_counts())