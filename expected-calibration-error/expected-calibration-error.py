import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=float)
    n = len(y)

    # handle p = 1.0 case, assign to last bin by np.floor
    eps = 1e-9
    p[p == 1.0] = p[p == 1.0] - eps
    
    bins = np.floor(p * n_bins)

    errors = []
    for m in range(n_bins):
        acc_m = np.mean(y[bins == m])
        conf_m = np.mean(p[bins == m])
        bins_m_len = len(bins[bins == m])

        error = bins_m_len * np.abs(acc_m - conf_m) / n
        # print(bins_m_len, acc_m, conf_m, error)
        errors.append(error)

    ece = np.nansum(errors)
    return float(ece)