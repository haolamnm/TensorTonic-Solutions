import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    mean = np.mean(x)
    med = np.median(x)
    mode, _ = Counter(x).most_common()[0]

    return mean, med, mode