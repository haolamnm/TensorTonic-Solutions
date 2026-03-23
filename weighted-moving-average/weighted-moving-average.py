def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    w_len = len(weights)
    v_len = len(values)
    wma = []
    for i in range(v_len - w_len + 1):
        s = sum(v * w for v, w in zip(values[i:i+w_len], weights))
        s /= sum(weights)
        wma.append(s)

    return wma