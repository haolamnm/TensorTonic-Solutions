import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    fn = {
        "mean": np.mean,
        "sum": np.sum
    }
    
    a, b, y = np.atleast_2d(a), np.atleast_2d(b), np.asarray(y)
    d = np.linalg.norm(a - b, axis=1)
    l = y * d**2 + (1 - y) * (np.maximum(0, margin - d)**2)

    return fn[reduction](l)