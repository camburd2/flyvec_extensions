import utils
import json
import encoder


def load_vocab(filepath):
    with open(filepath, 'r') as file:
        embeddings = json.load(file)
    vocab = {word: idx for idx, word in enumerate(embeddings.keys())}
    return vocab


def create_embeddings(model, vocab, hash_length):
    """Create dict to map words in vocab to binary embeddings from the model."""
    
    enc = encoder.Encoder(vocab=vocab)
    word_embeddings = {}

    for word, id in vocab.items():
        one_hot = enc.one_hot(id)
        embedding = model.get_embedding(x=one_hot, hash_len=hash_length)
        word_embeddings[word] = embedding.int().tolist()
    
    return word_embeddings


if __name__ == "__main__":

    # Load model and vocab
    checkpoint = 'model_checkpoint_17pct'
    model = utils.load_model(f'trained_models/openwebtext_checkpoints/{checkpoint}.pt')
    vocab = load_vocab('simple-flyvec-embeddings.json')

    # Get embeddings
    hash_length = 70
    embeddings = create_embeddings(model, vocab, hash_length)

    # Save embeddings
    save_path = f'embeddings/hash{hash_length}_{checkpoint}.json'
    with open(save_path, 'w') as file:
        json.dump(embeddings, file)

        print(f'embeddings created and saved:\t{save_path}')
