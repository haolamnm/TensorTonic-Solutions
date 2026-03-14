import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float) # (N, D)
    N, D = X.shape
    w = np.zeros(D) # (D,)
    b = 0.0

    for i in range(steps):
        # print(f"Iteration #{i}:")
        p = _sigmoid(X @ w + b)

        L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        dw = X.T @ (p - y) / N
        db = np.mean(p - y)

        w -= dw * lr
        b -= db * lr

    return w, b
        