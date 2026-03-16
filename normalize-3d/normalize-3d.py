import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.atleast_2d(v)
    distances = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
    results = v.T / distances
    results = results.T
    return results.squeeze() if len(results) == 1 else results