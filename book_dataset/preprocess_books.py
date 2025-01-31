import json
import re
from collections import Counter
from nltk.corpus import stopwords
import numpy as np

def load_book_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        
    # join all books into a string
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
    
    return np.array(processed_text), final_counts, vocab


def clean_save_data():
    # load data and combine books into a string
    print('loading and combining book data')
    combined_books_text = load_book_data(filepath='train.json')

    # clean/filter text
    print('clean/filter text')
    words_list, word_counts, vocab = preprocess_text(combined_books_text)

    # save book data
    np.save('book_dataset/data/book_words_array.npy', words_list) 

    with open('book_dataset/data/book_word_counts.json', 'w') as f:
        json.dump(word_counts, f)

    with open("book_dataset/data/book_vocab.json", "w") as f:
        json.dump(vocab, f)


def load_processed(train_window_size):

    with open("book_dataset/data/book_vocab.json", "r") as f:
        vocab = json.load(f)
    with open("book_dataset/data/book_word_counts.json", "r") as f:
        word_counts = json.load(f)

    words_arr = np.load("book_dataset/data/book_words_array.npy", allow_pickle=True)

    words_arr = words_arr[:len(words_arr) - len(words_arr) % train_window_size]
    train_data = words_arr.reshape(-1, train_window_size)

    return train_data, vocab, word_counts


if __name__ == "__main__":
    #clean_save_data()

    train_book, vocab, _ = load_processed(train_window_size=10)

    print(f'train data: shape = {train_book.shape}\ntrain sample  {train_book[0]}')
