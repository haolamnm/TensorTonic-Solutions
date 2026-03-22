import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.asarray(Q, dtype=float)
    Q_new = Q.copy()

    target = r + gamma * np.max(Q_new[s_next, :])

    Q_new[s, a] = Q_new[s, a] + alpha * (target - Q_new[s, a])
    return Q_new
