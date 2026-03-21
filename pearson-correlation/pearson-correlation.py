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
    sigma = np.sqrt(np.diag(X_cov))
    std_devs = np.outer(sigma, sigma)

    if not std_devs.all():
        X_cov = np.full((D, D), np.nan)
        mask = std_devs.astype(bool)
        X_cov[mask] = 1
        return X_cov 
    
    return X_cov / std_devs