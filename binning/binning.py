import numpy as np

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    values = np.asarray(values)

    min_val = min(values)
    max_val = max(values)
    w  = (max_val - min_val) / num_bins

    if (w == 0):
        # if all values are identical
        bins = np.full(len(values), 0)
    else:
        # otherwise
        bins = np.minimum((values - min_val) // w, num_bins - 1)
    
    return bins.tolist()