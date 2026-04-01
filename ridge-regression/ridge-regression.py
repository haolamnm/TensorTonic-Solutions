import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    _, d = X.shape
    print(d)
    I = np.eye(d)

    Z = X.T @ X + lam * I
    Z_inv = np.linalg.inv(Z)

    w = Z_inv @ X.T @ y
    return w.tolist()