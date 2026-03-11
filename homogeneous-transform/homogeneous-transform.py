import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.asarray(points)
    T = np.asarray(T)
    squeezed = points.ndim == 1
    points = np.atleast_2d(points) # (3,) -> (1, 3)
    
    # Turn (N, 3) -> (N, 4)
    ones = np.ones((points.shape[0], 1))
    points = np.hstack((points, ones))

    # Beautiful slicing
    results = (T @ points.T).T[:, :3]

    return results.squeeze() if squeezed else results
