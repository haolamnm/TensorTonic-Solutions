import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    V = np.asarray(V, dtype=float)
    sigma = r + gamma * V[s_next] - V[s]
    V[s] = V[s] + alpha * sigma

    return V