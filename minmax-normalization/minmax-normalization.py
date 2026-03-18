import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X)
    X_min = np.min(X, axis=axis)
    X_max = np.max(X, axis=axis)
    print("max", X_max)
    print("min", X_min)
    if X.ndim == 2 and axis == 1:
        X_new = (X.T - X_min) / (X_max - X_min + eps)
        X_new = X_new.T
    else:
        X_new = (X - X_min) / (X_max - X_min + eps)
    print("new", X_new)
    return X_new