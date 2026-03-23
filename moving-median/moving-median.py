def median(values):
    n = len(values)
    is_even = int(n % 2 == 0)
    mid_l = n // 2 - is_even
    mid_h = n // 2
    view = sorted(values)

    return (view[mid_l] + view[mid_h]) / 2

def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    medians = []
    n = len(values)
    for i in range(n - window_size + 1):
        slice = values[i:i+window_size]
        med = median(slice)
        medians.append(med)

    return medians