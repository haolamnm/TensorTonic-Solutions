import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)
    zeros = np.zeros(x.shape)

    return np.maximum(x, zeros)