import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k = np.asarray(k)
    return (1 - p)**(k - 1) * p, 1 / p
    