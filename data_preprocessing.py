import pandas as pd
import numpy as np
from joblib import dump
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Custom words to be removed
stop_words_manual = ['also', 'get', 'causing']

def remove_irrelevant_words(text):
    """
    Removes custom defined words from the text.
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words_manual]
    return ' '.join(filtered_words)

# Word count and tokenizer creation
def create_tokenizer(X_train):
    """
    Creates a tokenizer based on the frequency of words in the training data.
    """
    word_freq = Counter()
    for text in X_train['text']:
        word_freq.update(text.split())

    tokenizer = {word: i for i, (word, _) in enumerate(word_freq.most_common(200), start=1)}
    return tokenizer

# Tokenization function
def apply_tokens(text, tokenizer):
    """
    Tokenizes the text using the provided tokenizer.
    """
    tokens = text.split() if isinstance(text, str) else text
    return [tokenizer.get(token, 0) for token in tokens]

# Padding function
def apply_padding(text, max_length):
    """
    Applies padding to the text to make all sequences have the same length.
    """
    return [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq for seq in text]

# One-Hot Encoding function
def one_hot_encode(text, num_words):
    """
    Applies one-hot encoding to the text.
    """
    one_hot_encoded = np.zeros((len(text), num_words))
    for i, seq in enumerate(text):
        for idx in seq:
            if 0 < idx < num_words:
                one_hot_encoded[i, idx - 1] = 1
    return one_hot_encoded

# Label encoding
def encode_labels(y_train):
    """
    Encodes the labels using LabelEncoder.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train)
    label_names = {label: i for i, label in enumerate(label_encoder.classes_)}
    return encoded_labels, label_names