import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        return None

    n = len(y_true)
    loss = y_pred[np.arange(n), y_true]
    return -np.mean(np.log(loss))