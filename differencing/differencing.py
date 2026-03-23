import numpy as np

def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    series = np.asarray(series, dtype=float)

    view = series
    for d in range(order):
        n = len(view)
        first = view[:n-1]
        second = view[1:]
        view = second - first
        print(view)

    return view.tolist()