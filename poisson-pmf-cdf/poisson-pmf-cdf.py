import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    indices = np.arange(k+1)
    factorial = np.vectorize(math.factorial)
    probs = np.exp(-lam) * np.power(lam, indices) / factorial(indices)

    return probs[k], probs.sum()