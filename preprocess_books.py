import json
import re
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import random
import torch

def load_book_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    text = ' '.join(book['EN'] for book in data)
    return text


def preprocess_text(text, min_freq=50):
    # Clean
    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    # Get counts and filter out rare words
    counts = Counter(words)
    frequent_words = {word for word, count in counts.items() if count > min_freq}
    
    # Replace rare words with <unk> and create final vocab
    processed_text = ['<unk>' if word not in frequent_words else word for word in words]
    final_counts = Counter(processed_text)
    vocab = {word: idx for idx, word in enumerate(final_counts)}
    
    return processed_text, final_counts, vocab


def prepare_training_data(words, window_size):
    # Trim to full windows
    words_arr = np.array(words)
    words_arr = words_arr[:len(words_arr) - len(words_arr) % window_size]
    
    # Reshape into windows
    return words_arr.reshape(-1, window_size)
