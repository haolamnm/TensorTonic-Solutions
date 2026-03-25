import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    if w_norm == 0 or v_norm == 0:
        return np.nan

    cos_theta = np.dot(v, w) / (v_norm * w_norm)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    return theta