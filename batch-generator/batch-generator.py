import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    if not rng:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N = X.shape[0]
    indices = np.arange(N)
    rng.shuffle(indices)
    
    for i in range(0, N, batch_size):
        view = indices[i:i+batch_size]
        
        if drop_last and len(view) < batch_size:
           break

        X_batch, y_batch = X[view], y[view]
        yield (X_batch, y_batch)