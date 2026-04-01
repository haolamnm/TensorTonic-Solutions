import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    l, r = 0, 1
    y_l = y[split_mask == l]
    y_r = y[split_mask == r]

    n_l = len(y_l)
    n_r = len(y_r)
    n = n_l + n_r

    ig = _entropy(y) - (n_l / n * _entropy(y_l) + n_r / n * _entropy(y_r))
    return ig
