import numpy as np

def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    priorities = np.asarray(priorities, dtype=float)
    powered_priorities = priorities ** alpha
    print(powered_priorities)

    probs = powered_priorities / np.sum(powered_priorities)
    print(probs)

    batch_size = len(probs)
    weights = (batch_size * probs) ** (-beta)
    print(weights)

    weights_hat = weights / np.max(weights)
    print(weights_hat)

    return [probs.tolist(), weights_hat.tolist()]
    