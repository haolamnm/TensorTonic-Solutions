import numpy as np

def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    rng = np.random.RandomState(seed=seed)
    buffer = np.asarray(buffer, dtype=float)

    # sampling
    n = len(buffer)
    indices = np.arange(n)
    picked = rng.choice(indices, size=batch_size, replace=False)
    print(picked)
    
    return buffer[picked]