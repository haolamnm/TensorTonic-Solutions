def interp(i, left, v_left, right, v_right):
    delta = v_right - v_left
    return v_left + delta * (i - left) / (right - left)
    

def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    results = []
    n = len(values)
    for i, v in enumerate(values):
        if v is not None:
            results.append(v)
            continue

        left, right = 0, n
        for l in range(max(i - 1, 0), -1, -1):
            if values[l] is not None:
                left = l
                break

        for r in range(min(i + 1, n), n, 1):
            if values[r] is not None:
                right = r
                break
                
        value = interp(i, left, values[left], right, values[right])
        results.append(value)

    return results