import numpy

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # all values are non-negative
    x = np.asarray(values, dtype=float)
    y = np.log(1 + x)
    
    return y