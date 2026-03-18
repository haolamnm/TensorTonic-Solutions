import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)
    nuy = np.mean(X, axis=axis, keepdims=True, dtype=float)
    sig = np.std(X, axis=axis, keepdims=True, dtype=float)
    
    Z = (X - nuy) / (sig + eps)
    return Z
