import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    predictions = np.asarray(predictions, dtype=float)
    K = len(predictions)

    scaled_epsilon = epsilon / K
    q = np.where(np.arange(K) == target,
                (1.0 - epsilon) + scaled_epsilon,
                 scaled_epsilon
                )
    
    L = -np.sum(q * np.log(predictions))
    return L