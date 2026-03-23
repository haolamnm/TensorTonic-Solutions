def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    ema = [values[0]]

    for t in range(1, len(values)):
        ema.append(alpha * values[t] + (1 - alpha) * ema[t - 1])

    return ema