def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    if not rewards:
        return []

    n = len(rewards)
    G = [0] * n
    G[-1] = rewards[-1]

    # cleanly walking backward
    # backward recursive relation
    for t in range(n - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    return G