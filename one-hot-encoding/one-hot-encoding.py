import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y, dtype=int)
    detected_classes = np.max(y) + 1
    if num_classes is None or num_classes < detected_classes:
        num_classes = detected_classes

    N = len(y)
    indices = np.arange(N)
    results = np.zeros((N, num_classes))
    results[indices, y] = 1

    return results
