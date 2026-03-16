import numpy as np
from scipy.special import comb


def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    
    def prob(n, p, k):
        return comb(n, k) * (p**k) * ((1 - p)**(n - k))

    probs = (prob(n, p, i) for i in range(k+1))

    return prob(n, p, k), sum(probs)