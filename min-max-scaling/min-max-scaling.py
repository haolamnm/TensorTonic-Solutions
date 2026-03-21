import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    x = np.asarray(data, dtype=float)
    max_vals = np.max(x, axis=0)
    min_vals = np.min(x, axis=0)

    mask = max_vals == min_vals
    mask = np.broadcast_to(mask, x.shape)

    eps = 1e-12
    x_new = (x - min_vals) / (max_vals - min_vals + eps)
    x_new[mask] = 0.0

    return x_new.tolist()