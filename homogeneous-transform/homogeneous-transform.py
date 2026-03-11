import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.asarray(points)
    T = np.asarray(T)

    flag = False
    if points.ndim == 1: # check for N=1 3D point
        flag = True
        points = np.expand_dims(points, axis=0) # (1, 3)

    DIM = points.shape[1]
    
    # Turn (N, 3) -> (N, 4)
    ones = np.ones((points.shape[0], 1))
    points = np.hstack((points, ones))

    new_points = (T @ points.T).T

    # Drop final col
    results = np.delete(new_points, DIM, axis=1)

    if flag:
        results = results.reshape(DIM)

    return results
