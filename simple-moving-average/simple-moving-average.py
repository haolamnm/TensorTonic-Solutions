def mean(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    n = len(values)
    means = []
    for i in range(n - window_size + 1):
        means.append(mean(values[i:i+window_size]))

    return means