import numpy as np

def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    matrix = np.asarray(matrix, dtype=float)

    # store un-rate slots
    mask = matrix == 0
    print(mask)
    
    counts = np.count_nonzero(matrix, axis=1, keepdims=True)
    sums = np.sum(matrix, axis=1, keepdims=True)
    means = sums / counts
    results = matrix - means

    results[mask] = 0.0
    
    return results.tolist()