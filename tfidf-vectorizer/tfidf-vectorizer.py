import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # build vocab
    tokens = [doc.lower().split() for doc in documents]
    vocab = sorted(set(word for doc in tokens for word in doc))
    word2idx = {word: i for i, word in enumerate(vocab)}

    # calc tf
    n_docs = len(documents)
    n_vocab = len(vocab)
    tf = np.zeros((n_docs, n_vocab))

    for i, doc in enumerate(tokens):
        counts = Counter(doc)
        for word, count in counts.items():
            tf[i, word2idx[word]] = count / len(doc)

    # calc idf
    idf = np.zeros(n_vocab)

    for j, word in enumerate(vocab):
        df = sum(1 for doc in tokens if word in doc)
        idf[j] = math.log(n_docs / df)

    # compute final tf-idf
    tfidf = tf * idf

    return tfidf, vocab


    