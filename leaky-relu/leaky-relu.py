import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x_np = np.asarray(x)
    return np.where(x_np >= 0, x_np, x_np * alpha)