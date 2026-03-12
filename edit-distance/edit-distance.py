def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # Check for shorter string
    # s1 should be the longer string
    if len(s1) < len(s2):
        s1, s2 = s2, s1 # swap
    
    # Dynamic Programming
    dp = [list(range(len(s1) + 1))]
    for i in range(1, len(s2) + 1):
        dp.append([0] * (len(s1) + 1))
        dp[i][0] = i

    for j in range(1, len(s1) + 1):
        for i in range(1, len(s2) + 1):
            if s1[j - 1] == s2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                delete = dp[i - 1][j]
                insert = dp[i][j - 1]
                replace = dp[i - 1][j - 1]
                dp[i][j] = 1 + min(delete, insert, replace)

    return dp[len(s2)][len(s1)]