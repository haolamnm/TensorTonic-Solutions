import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.atleast_2d(np.asarray(v, dtype=float))

    norm = np.sqrt(np.sum(np.square(v), axis=1))
    
    return float(norm.squeeze(axis=0)) if norm.shape[0] == 1 else norm