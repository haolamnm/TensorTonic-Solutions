def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    counts = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1

    freqs = [counts[val] / len(values) for val in values]
    return freqs