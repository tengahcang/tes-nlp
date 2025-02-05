import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import re

df = pd.read_csv('preprocessed_text_output.csv')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

all_words = []

def tokenize(text):
    return re.findall(r'\w+', text)

for index, row in df.iterrows():
    tokens = tokenize(row['preprocessed_text'])
    all_words.extend(tokens)

word_freq = Counter(all_words)

most_common_words = word_freq.most_common()

df_word_freq = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

df_word_freq.to_csv('word_frequencies.csv', index=False)

print("Word frequencies saved to 'word_frequencies.csv'")
