def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    result = []
    for p in range(period):
        i = 0
        view = []
        while p + i * period < len(series):
            index = p + i * period
            view.append(series[index])
            i += 1
        result.append(sum(view) / len(view))

    return result
        