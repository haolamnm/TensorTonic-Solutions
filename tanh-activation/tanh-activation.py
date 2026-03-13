import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype=float)
    pos = np.exp(x)
    neg = np.exp(-x)

    return (pos - neg) / (pos + neg)