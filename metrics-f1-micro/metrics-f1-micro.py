import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    classes = len(set(y_true) | set(y_pred))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp, fp, fn = 0, 0, 0
    for c in range(classes):
        tp += np.sum(np.logical_and(y_true == c, y_pred == c))
        fp += np.sum(np.logical_and(y_true != c, y_pred == c))
        fn += np.sum(np.logical_and(y_true == c, y_pred != c))

    if tp + fp + fn == 0:
        return 1.0
    
    score = 2 * tp / (2 * tp + fp + fn)
    return score

