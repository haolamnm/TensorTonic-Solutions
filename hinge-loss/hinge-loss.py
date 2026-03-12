import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    fn = {
        "mean": np.mean,
        "sum": np.sum
    }
    if reduction not in fn:
        raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    
    losses = np.maximum(0, margin - y_true * y_score)
    return fn[reduction](losses)
    