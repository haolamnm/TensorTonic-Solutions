import numpy as np

def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    lags = np.asarray(lags, dtype=int)
    series = np.asarray(series, dtype=float)
    rows = []

    max_lag = np.max(lags)

    for t in range(max_lag, len(series)):
        row = series[t - lags]
        rows.append(row.tolist())

    return rows