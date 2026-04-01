import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    y = np.asanyarray(y_true)
    p = np.asanyarray(y_pred)
    n = len(y)

    bins = np.clip(p * n_bins, 0, n_bins - 1).astype(int)

    # only care about bins that contain sth
    bin_counts = np.bincount(bins, minlength=n_bins)
    nonzero = bin_counts > 0
    
    # sum of y (accuracy numerator) and p (confidence numerator) per bin
    bin_acc_sum = np.bincount(bins, weights=y, minlength=n_bins)
    bin_conf_sum = np.bincount(bins, weights=p, minlength=n_bins)

    # only compute for bins with data to avoid divide-by-zero
    acc_m = bin_acc_sum[nonzero] / bin_counts[nonzero]
    conf_m = bin_conf_sum[nonzero] / bin_counts[nonzero]
    
    # weight each bin by its size relative to total N
    ece = np.sum(bin_counts[nonzero] * np.abs(acc_m - conf_m) / n)
    
    return float(ece)