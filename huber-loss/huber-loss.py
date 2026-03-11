import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    error = y_true - y_pred

    loss = np.where(np.abs(error) <= delta,
                    0.5 * np.square(error),
                    delta * (np.abs(error) - 0.5 * delta)
                   )
    return np.mean(loss)