import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=float)

    # Best practice for big positive x
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (np.exp(x) + 1)
                   )