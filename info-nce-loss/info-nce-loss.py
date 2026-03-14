import numpy as np

def stable_softmax(X):
    # [min(X), max(X)]
    # [min(X) - max(X), 0]
    # exp([min(X) - max(X), 0]) -> [0, 1]
    Z = np.exp(X - np.max(X, axis=1, keepdims=True))
    N, _ = Z.shape
    P = Z / np.sum(Z, axis=1, keepdims=True)

    # scores -> probs
    return P

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    N, D = Z1.shape

    S = (Z1 @ Z2.T) / temperature

    # take only (i, i) pair
    P = np.diag(stable_softmax(S))
    P_safe = np.clip(P, 1e-12, 1.0)

    L = -np.mean(np.log(P))
    return L