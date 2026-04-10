def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0

    for t in reversed(range(n)):
        delta = rewards[t] + gamma * values[t+1] - values[t]

        advantages[t] = delta + gamma * lam * last_gae
        last_gae = advantages[t]

    return advantages