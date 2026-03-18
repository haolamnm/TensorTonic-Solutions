import numpy as np

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    L = (6 / (fan_in + fan_out))**0.5

    W = np.asarray(W, dtype=float)
    W_new = (W * 2 * L) - L
    return W_new