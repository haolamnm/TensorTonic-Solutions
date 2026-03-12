def edit_distance(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1 # s2 shorter than s1

    prev = list(range(len(s1) + 1))

    for i in range(1, len(s2) + 1):
        curr = [i] + [0] * len(s1)

        for j in range(1, len(s1) + 1):
            if s2[i - 1] == s1[j - 1]:
                curr[j] = prev[j - 1]
            else:
                delete = prev[j]
                insert = curr[j - 1]
                replace = prev[j - 1]
                curr[j] = 1 + min(delete, insert, replace)

        prev = curr

    return prev[len(s1)]