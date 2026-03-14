import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=bool)

    losses = -np.pow((1 - p), gamma) * y * np.log(p) - np.pow(p, gamma) * (1 - y) * np.log(1 - p)

    return np.mean(losses)