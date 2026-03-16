import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.atleast_2d(X)
    N, D = X.shape
    if N <= 1:
        return None
    
    m = np.mean(X, axis=0)
    Xc = X - m
    cov = 1 / (N - 1) * (Xc.T @ Xc)
    return cov