import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    base = np.linalg.norm(x1) * np.linalg.norm(x2)
    cos = np.dot(x1, x2) / base

    if label == 1:
        return 1 - cos
    else: # Label is either 1 or -1
        return max(0, cos - margin)
