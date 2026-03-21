import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype=float) # (N, D)
    N, D = X.shape
    if X.ndim != 2 or N < 2:
        return None

    X_m = X - np.mean(X, axis=0, keepdims=True)
    X_cov = (X_m.T @ X_m) / (N - 1)
    std = np.std(X, axis=0, ddof=1)
    std_devs = np.outer(std, std)

    with np.errstate(invalid="ignore"):
        return X_cov / std_devs