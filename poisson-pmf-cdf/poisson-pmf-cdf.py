import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    indices = np.arange(k+1)

    # log(k!) = sum(log(1), log(2), ..., log(k))
    log_factorial = np.zeros(k + 1)
    if k > 0:
        log_factorial[1:] = np.cumsum(np.log(indices[1:]))

    log_probs = -lam + indices * np.log(lam) - log_factorial
    probs = np.exp(log_probs)

    return probs[k], probs.sum()