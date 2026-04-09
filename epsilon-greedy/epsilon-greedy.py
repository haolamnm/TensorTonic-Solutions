import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values))) # explore
    else:
        return int(np.argmax(q_values))