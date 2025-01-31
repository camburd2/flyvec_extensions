from tqdm import tqdm
import nltk
import book_dataset.preprocess_books as prep
import utils
from model import FlyvecModel


def train_book(train_data, num_epochs, encoder):
    """train model on book data
    
    Args:
        train_data (np array): shape = (N, window size)
        num_epochs
        encoder (utils.Encoder()): encoder for book dataset
    """
    
    for epoch in range(num_epochs):
        for sample in tqdm(train_data, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100, leave=True):        
            enc_sample = encoder.one_hot(sample)
            model.update(enc_sample)


if __name__ == "__main__":
    nltk.download('stopwords') 

    # load preprocessed book data
    train_data_book, vocab_book, _ = prep.load_processed(train_window_size=10)
    print(f'train data: shape = {train_data_book.shape}\ntrain sample  {train_data_book[0]}')

    # initialize model and encoder
    model = FlyvecModel(
        K_size=350,                     # Number of neurons
        vocab_size=len(vocab_book),     # Size of vocab
        k=5,                            # Update top-k neurons
        lr=.1,                          # Learning rate
        norm_rate=5,                    # Normalization rate
    )
    enc = utils.Encoder(vocab=vocab_book)

    train_book(train_data=train_data_book, num_epochs=1, encoder=enc)