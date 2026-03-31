import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """ 
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 and x.ndim != 4:
        raise ValueError
    
    x_pooled = np.mean(x, axis=(-1, -2))

    return x_pooled