import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    targets = np.asarray(targets, dtype=bool)
    predictions = np.asarray(predictions, dtype=float)
    
    pt = np.abs(targets - (1 - predictions))
    losses = -alpha * np.pow((1 - pt), gamma) * np.log(pt)

    return np.mean(losses)