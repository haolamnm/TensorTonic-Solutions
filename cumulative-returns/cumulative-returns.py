import numpy as np

def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    returns = np.asarray(returns, dtype=float)
    cums = np.cumprod(1 + returns) - 1
    return cums.tolist()
