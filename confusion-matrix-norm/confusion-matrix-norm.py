import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    # confusion matrix
    C = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1

    # vectorized
    # np.add.at(C, (y_true, y_pred), 1)

    # normalization
    if normalize == 'true':
        row_sums = C.sum(axis=1, keepdims=True)
        C = C / np.where(row_sums == 0, 1, row_sums)
    elif normalize == 'pred':
        col_sums = C.sum(axis=0, keepdims=True)
        C = C / np.where(col_sums == 0, 1, col_sums)
    elif normalize == 'all':
        C = C / C.sum()

    return C