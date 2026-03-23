def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    W = [1]
    cums = []
    for t, r in enumerate(returns):
        W.append((1 + r) * W[t])
        cums.append(W[t + 1] - 1)

    return cums