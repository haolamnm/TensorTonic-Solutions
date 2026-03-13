import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x)

    # numerically stable sigmoid
    sigmoid = np.where(x >= 0,
                      1 / (1 + np.exp(-x)),
                      np.exp(x) / (np.exp(x) + 1))
    return x * sigmoid