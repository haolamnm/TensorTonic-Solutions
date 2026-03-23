import numpy as np

def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    X = np.asarray(values, dtype=float)
    return (X[:, None] ** np.arange(degree + 1)).tolist()
