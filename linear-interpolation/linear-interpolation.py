def interp(i, left, v_left, right, v_right):
    delta = v_right - v_left
    return v_left + delta * (i - left) / (right - left)
    

def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    n = len(values)
    results = list(values)

    i = 0
    while i < n:
        if results[i] is not None:
            i += 1
            continue

        # find the right anchor
        r = i + 1
        while r < n and results[r] is None:
            r += 1

        # find the left anchor
        # clean
        l = i - 1 
        
        # fill the gap [i, r)
        for j in range(i, r):
            if l < 0:
                results[j] = results[r] # clamp left edge
            elif r >= n:
                results[j] = results[l] # clamp right edge
            else:
                results[j] = interp(j, l, results[l], r, results[r])

        i = r

    return results