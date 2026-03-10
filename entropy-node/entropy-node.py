import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0

    _, cnts = np.unique(y, return_counts=True)
    probs = cnts / len(y)
    return -np.sum(probs * np.log2(probs))
    