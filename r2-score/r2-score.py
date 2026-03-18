import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.mean(y_true, dtype=float)

    if (len(set(y_true)) == 1):
        return (y_pred == y_true).all()

    ssr = np.sum(np.square(y_true - y_pred))
    sst = np.sum(np.square(y_true - y_mean))

    return 1.0 - ssr / sst