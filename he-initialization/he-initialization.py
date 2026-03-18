import numpy as np

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    L = (6 / fan_in)**0.5
    W = np.asarray(W, dtype=float)
    W_scaled = (W * 2 * L) - L
    return W_scaled