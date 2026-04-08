def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    levels = [series[0]]
    trends = [series[1] - series[0]]

    for t in range(1, len(series)):
        level = alpha * series[t] + (1 - alpha) * (levels[t - 1] + trends[t - 1])
        trend = beta * (level - levels[t - 1]) + (1 - beta) * trends[t - 1]

        levels.append(level)
        trends.append(trend)

    return levels