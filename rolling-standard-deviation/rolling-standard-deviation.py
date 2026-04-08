def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    sigmas = []
    for i in range(len(values) - window_size + 1):
        slice = values[i : i + window_size]
        mu = sum(slice) / len(slice)
        sigma = (sum((x - mu)**2 for x in slice) / len(slice))**0.5

        sigmas.append(sigma)

    return sigmas