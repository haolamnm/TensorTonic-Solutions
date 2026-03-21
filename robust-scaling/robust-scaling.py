import numpy as np


def tukey_quartiles(values):
    values = np.sort(values)
    n = len(values)
    mid = n // 2
    lower_half = values[:mid]
    upper_half = values[mid + n % 2:]
    return np.median(lower_half), np.median(upper_half)

def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    # check for single element
    if len(values) <= 1:
        return [0]

    x = np.asarray(values, dtype=float)
    q1, q3 = tukey_quartiles(x)
    print(q1, q3)
    med = np.median(x)
    print(med)
    x_scaled = (x - med) / ((q3 - q1) or 1)

    return x_scaled