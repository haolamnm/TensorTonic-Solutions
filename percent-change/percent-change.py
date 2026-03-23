def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    pct = [0] * (len(series) - 1)
    for i in range(1, len(series)):
        prev = series[i - 1]
        diff = series[i] - series[i - 1]
        pct[i - 1] = diff / prev if prev != 0 else 0

    return pct