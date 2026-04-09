def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    n = len(series)
    mean = sum(series) / n
    gamma_0 = sum((x - mean) ** 2 for x in series)
    
    # edge case where all values are identical (zero variance)
    if gamma_0 == 0:
        return [1.0] + [0.0] * max_lag
        
    results = []
    
    for k in range(max_lag + 1):
        autocovariance = 0
        for t in range(n - k):
            autocovariance += (series[t] - mean) * (series[t + k] - mean)
        r_k = autocovariance / gamma_0
        results.append(r_k)
        
    return results