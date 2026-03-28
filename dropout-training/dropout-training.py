import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)
    
    if rng is None:
        rng = np.random.default_rng()
    
    keep = rng.random(x.shape) < (1 - p)
    mask = np.where(keep, 1 / (1 - p), 0).astype(x.dtype)
    
    return x * mask, mask