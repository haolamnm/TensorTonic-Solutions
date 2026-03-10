import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    
    cls, cnts = np.unique(y, return_counts=True)
    probs = cnts / len(y) # normalize
    
    entropy = 0.0
    for i in range(len(cls)):
        if probs[i] == 0:
            continue
        entropy += probs[i] * np.log2(probs[i])
    return -entropy
    