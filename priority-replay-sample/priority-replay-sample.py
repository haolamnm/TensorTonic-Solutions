import numpy as np

def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    priorities = np.asarray(priorities, dtype=float)
    powered_priorities = priorities ** alpha

    probs = powered_priorities / np.sum(powered_priorities)

    batch_size = len(probs)
    weights = (batch_size * probs) ** (-beta)

    weights_hat = weights / np.max(weights)

    return [probs.tolist(), weights_hat.tolist()]
    