import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # pred (N, D)
    pred = np.asarray(predictions, dtype=int)
    N, D = pred.shape
    K = np.max(pred) + 1

    # clever shifting
    offsets = np.arange(D) * K
    pred_shifted = pred + offsets
    # print(pred_shifted)

    counts = np.bincount(pred_shifted.ravel(), minlength=D*K).reshape(D, K)
    picked = np.argmax(counts, axis=1)
    
    return picked.tolist()

    