import numpy as np

def normalize(matrix, axis, norm_type):
    if norm_type == "l2":
        return np.sqrt(np.sum(np.square(matrix), axis=axis, keepdims=True))
    if norm_type == "l1":
        return np.sum(np.abs(matrix), axis=axis, keepdims=True)
    if norm_type == "max":
        return np.max(matrix, axis=axis, keepdims=True)
    else:
        raise ValueError
    
def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        return None

    # Matrix (N, M)
    # axis=None:  (1, 1)
    # axis=0   :  (1, M)
    # axis=1   :  (N, 1)
    try:
        norm = normalize(matrix, axis, norm_type)
        norm = np.broadcast_to(norm, matrix.shape)
    except ValueError:
        return None
    
    # Matrix (N, M) / norm
    eps = 1e-12
    return matrix / (norm + eps)
    