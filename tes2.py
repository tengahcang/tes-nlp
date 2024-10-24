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
import gdown
import json


# Uncomment this line if you haven't downloaded NLTK stopwords yet
# nltk.download('stopwords')

# Function to download and load slang dictionary from Google Drive
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

# Adding more slang words to the dictionary
new_slang_dict = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "kagak": "tidak",
    "enggak": "tidak",
}

# Example of updating slang dictionary
slang_dict = new_slang_dict  # Assuming you have downloaded and loaded it using the function

# Function to normalize slang words in a text
def normalize_slang(text, slang_dict):
    words = word_tokenize(text)
    normalized_words = [slang_dict.get(word, word) for word in words]  # Replace slang if found in dictionary
    return ' '.join(normalized_words)

# Text preprocessing function including tokenization, lowercasing, punctuation removal, stopword removal, slang normalization, and stemming
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

# Create a stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Example text for preprocessing
text = "adlh bgmn nih? afaik ini bagus abis! daripada ga dikerjai sendiri"

# Preprocess the text
preprocessed_text = preprocess_text(text, slang_dict, stemmer)

# Print the final preprocessed text
print(preprocessed_text)
