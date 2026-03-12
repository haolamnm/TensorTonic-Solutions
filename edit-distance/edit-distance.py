def edit_distance(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # s2 is shorter -> fewer iterations

    prev = list(range(len(s1) + 1))

    for i in range(1, len(s2) + 1):
        curr = [i] + [0] * len(s1)
        for j in range(1, len(s1) + 1):
            if s2[i - 1] == s1[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[len(s1)]