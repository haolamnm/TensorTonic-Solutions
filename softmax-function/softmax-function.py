import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.atleast_2d(x)

    numerator = np.exp(x - np.max(x))
    denominator = np.sum(numerator, axis=1, keepdims=True)

    # nice use of keepdims=True, 
    # making (N, C), (N ,1) broadcast work
    results = numerator / denominator
    return np.atleast_1d(results.squeeze())