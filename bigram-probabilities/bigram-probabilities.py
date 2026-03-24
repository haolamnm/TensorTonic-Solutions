from collections import defaultdict

def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # count bigrams
    counts = defaultdict(int)
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] += 1
    counts = dict(counts)

    # build vocab
    vocab = set(tokens)
    V = len(vocab)

    # count unigrams
    unigram_counts = defaultdict(int)
    for (w1, w2) in counts:
        unigram_counts[w1] += counts[(w1, w2)]

    # smooth probs
    probs = {}
    for w1 in vocab:
        denom = unigram_counts[w1] + V
        for v in vocab:
            numerator = counts.get((w1, v), 0) + 1
            probs[(w1, v)] = numerator / denom

    return counts, probs
