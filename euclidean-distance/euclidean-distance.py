import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x, y = np.asarray(x), np.asarray(y)
    return np.linalg.norm(x - y)