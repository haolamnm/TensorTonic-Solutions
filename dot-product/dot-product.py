import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch. got x={x.shape} and y={y.shape}")
    return np.dot(x, y)