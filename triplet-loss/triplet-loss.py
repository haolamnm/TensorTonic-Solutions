import numpy as np

def distance(x, y):
    return np.sum(np.square(x - y), axis=1)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)

    dap = distance(anchor, positive)
    dan = distance(anchor, negative)
    losses = np.maximum(0.0, dap - dan + margin)
    return np.mean(losses)