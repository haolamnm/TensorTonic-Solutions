import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N)

    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)

    # elegant use of array_split
    folds = np.array_split(indices, k)

    result = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        result.append((train_idx.astype(int), val_idx.astype(int)))

    return result
