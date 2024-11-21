import numpy as np
import torch
from context_model import ContextModel


class Encoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def tokenize(self, input):
        # Array or string
        if isinstance(input, str):
            words = input.split()
        else: 
            words = input
        return [self.vocab[word] for word in words]

    def one_hot(self, input):
        tokens = self.tokenize(input)
        encoded = torch.zeros(self.vocab_size, dtype=torch.float32)
        encoded[tokens] = 1
        return encoded



def calc_sim(v1, v2):
    v1 = v1.cpu().numpy()
    v2 = v2.cpu().numpy()
    n11 = np.sum((v1 == 1) & (v2 == 1))
    n00 = np.sum((v1 == 0) & (v2 == 0))
    n = len(v1)
    similarity = (n11 + n00) / n
    return similarity


def calc_print_sim_words(vocab, word_counts, model, word, hash_len=50, top_N=15):
    enc = Encoder(vocab)

    enc_target_word = enc.one_hot(word)
    hash_target = model.get_embedding(enc_target_word, hash_len)

    sims = []

    # Calculate  similarity between target and the rest of vocab
    for word in vocab.keys():
        enc_word = enc.one_hot(word)
        hash_word = model.get_embedding(enc_word, hash_len)
        sim = calc_sim(hash_target, hash_word)
        sims.append((word, sim))

    # Sort by similarity score and get top N
    top_N = sorted(sims, key=lambda x: x[1], reverse=True)[:top_N]

    # Print results
    print(f"{'Word':<15} {'Similarity':<10} {'Frequency':<10}")
    print("-" * 35)
    for word, sim in top_N:
        print(f"{word:<15} {sim:>9.3f} {word_counts[word]:>10}")


def save_model(model, filepath):
    state = {
        'W': model.W,
        'k': model.k,
        'lr': model.lr,
        'norm_rate': model.norm_rate,
    }
    torch.save(state, filepath)


def load_model(filepath, device='cpu'):
    state = torch.load(filepath, map_location=device)
    K_size, vocab_size = state['W'].shape
    
    model = ContextModel(
        K_size=K_size,
        vocab_size=vocab_size,
        k=state['k'],
        lr=state['lr'],
        norm_rate=state['norm_rate'],
        device=device
    )
    model.W = state['W']
    return model