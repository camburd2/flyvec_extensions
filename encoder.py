import re
import torch
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    print("Downloading stopwords...")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

num_token = "<NUM>"
unk_token = "<UNK>"

# Precompiled regex
clean_pattern = re.compile(r"[^\w\s]")  # Remove punctuation
number_pattern = re.compile(r"\b\d+\b")  # Match numbers


class Encoder:
    def __init__(self, vocab):
        """
        Initialize the encoder with a vocabulary.
        Args:
            vocab (dict): A dictionary mapping words to unique IDs.
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def preprocess(self, text, remove_stopwords=True):
        """
        Preprocess a text by cleaning, replacing numbers, and removing stopwords.
        Args:
            text (str): Input text to preprocess.
            remove_stopwords (bool): Whether to remove stopwords.

        Returns:
            list: List of preprocessed tokens.
        """
        # Clean text: lowercase, remove punctuation, and replace numbers
        text = clean_pattern.sub(" ", text.lower())
        text = number_pattern.sub(num_token, text)

        # Tokenize
        tokens = text.split()

        # Remove stopwords
        if remove_stopwords:
            tokens = [w for w in tokens if w not in stop_words]

        return tokens

    def tokenize(self, input):
        """
        Tokenize input into vocabulary indices.
        Args:
            input (list): List of preprocessed tokens.

        Returns:
            list: List of vocabulary indices.
        """
        return [self.vocab.get(word, self.vocab[unk_token]) for word in input]

    def one_hot(self, tokens, create_target_vector=False):
        """
        One-hot encode tokens.
        Args:
            tokens (list): List of token indices.

        Returns:
            torch.Tensor: One-hot encoded vector of shape [vocab_size].
        """
        encoded = torch.zeros(self.vocab_size, dtype=torch.float32)
        encoded[tokens] = 1
        if create_target_vector:
            target = torch.zeros(self.vocab_size, dtype=torch.float32)
            target[tokens[len(tokens) // 2]] = 1
            encoded = torch.cat((encoded, target))
        return encoded