import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    if X_train.ndim != X_test.ndim:
        raise ValueError("dimension mismatch")
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1) # (N_train, 1)
        X_test = X_test.reshape(-1, 1)   # (N_test, 1)

    N_train, D = X_train.shape
    N_test, _ = X_test.shape

    # diff (N_test, N_train, D)
    diff = X_test[:, None, :] - X_train[None, :, :]

    # dist (N_test, N_train)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    print(dist)

    indices = np.argsort(dist, axis=1)
    k_safe = min(k, N_train)
    result = np.full((N_test, k), -1, dtype=int)
    result[:, :k_safe] = indices[:, :k_safe]

    return result