import numpy as np

def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    lags = np.asarray(lags, dtype=int)
    series = np.asarray(series, dtype=float)

    max_lag = np.max(lags) # L
    t = np.arange(max_lag, len(series)) # T

    rows = series[t[:, None] - lags[None, :]]
    return rows.tolist()