import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.asarray(A)
    if A.ndim == 2 and A.shape[0] != A.shape[1]:
        return None
    if np.linalg.det(A) == 0:
        return None
        
    return np.linalg.inv(A)
