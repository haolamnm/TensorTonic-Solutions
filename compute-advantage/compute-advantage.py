import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    n = len(rewards)
    G = [0] * n
    G[-1] = rewards[-1]

    # backward recursive relation
    for t in range(n - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    # vectorized
    G = np.asarray(G, dtype=float)
    V = np.asarray(V, dtype=float)
    S = np.asarray(states, dtype=int)

    T = np.arange(n)
    A = G[T] - V[S[T]]
    return A
