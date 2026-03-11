import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    
    if y_pred.shape != y_true.shape:
        return None

    return np.average(np.square(y_pred - y_true))
