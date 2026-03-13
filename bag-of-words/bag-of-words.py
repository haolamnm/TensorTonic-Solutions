import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vocab_lookup = {}
    for index, word in enumerate(vocab):
        vocab_lookup[word] = index

    token_counts = {}
    for tok in tokens:
        token_counts[tok] = token_counts.get(tok, 0) + 1

    vector = [0] * len(vocab)

    for tok in tokens:
        if tok not in vocab_lookup:
            continue
        vector[vocab_lookup[tok]] = token_counts.get(tok, 0)

    return np.asarray(vector, dtype=int)    