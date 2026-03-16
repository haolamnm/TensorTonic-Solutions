import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x, dtype=int)
    pmf = np.where(x, p, 1 - p)
    mean = p
    var = p * (1 - p)

    return (pmf, mean, var)