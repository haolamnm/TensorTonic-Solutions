def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a, b = set(set_a), set(set_b)

    intersection = a & b
    union = a | b

    return (len(intersection) / len(union)) if union else 0.0