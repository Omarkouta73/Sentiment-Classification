import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('data.csv', 
                   encoding='latin-1', header=None)

data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

label_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
data['label'] = data['target'].map(label_map)

# Keep only necessary columns
data = data[['text', 'label']]

import pandas as pd
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 1. Normalize
def normalize(df):
    df["text"] = df["text"].str.lower()
    return df

def remove_mentions(text):
    return ' '.join(word for word in text.split() if not word.startswith('@'))

# 2. Remove punctuation and numbers
def remove_punctuation_numbers(df):
    df["text"] = df["text"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
    return df

# 3. Remove stopwords
def remove_stopwords(df):
    stop = set(stopwords.words('english'))
    df["text"] = df["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    return df

# 4. Tokenize
def tokenize(df):
    df["text"] = df["text"].apply(word_tokenize)
    return df

# 5. Lemmatization (instead of stemming)
def lemmatize(df):
    lemmatizer = WordNetLemmatizer()
    # Join tokens back into a string after lemmatization
    df["text"] = df["text"].apply(lambda tokens: ' '.join([lemmatizer.lemmatize(word) for word in tokens]))
    return df

def preprocess(df):
    df = normalize(df)
    df["text"] = df["text"].apply(remove_mentions)
    df = remove_punctuation_numbers(df)
    df = remove_stopwords(df)
    df = tokenize(df)
    df = lemmatize(df)
    return df



data = preprocess(data)


data.to_feather('data.feather')