import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]
    s_sorted = y_score[desc_idx]

    tp = np.cumsum(y_sorted)
    fp = np.arange(1, len(y_true)+1) - tp
    
    p = y_true.sum()
    n = len(y_true) - p

    tpr = tp / p
    fpr = fp / n

    # Keep only the LAST index of each tied group
    # e.g. scores [0.8, 0.5, 0.5, 0.3]. keep indices [0, 2, 3]
    keep = np.concatenate([
        np.where(s_sorted[:-1] != s_sorted[1:])[0],
        [len(y_true) - 1]
    ])

    tpr = np.concatenate([[0.0], tp[keep] / p])
    fpr = np.concatenate([[0.0], fp[keep] / n])
    thresholds = np.concatenate([[np.inf], s_sorted[keep]])

    return fpr, tpr, thresholds
