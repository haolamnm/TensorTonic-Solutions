import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    targets = np.asarray(targets, dtype=bool)
    predictions = np.asarray(predictions, dtype=float)

    pt = np.where(targets, predictions, 1 - predictions)
    pt = np.clip(pt, 1e-7, 1.0)
    losses = -alpha * (1 - pt)**gamma * np.log(pt)

    return np.mean(losses)