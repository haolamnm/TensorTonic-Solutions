import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments)
    
    centroids = []
    fallback = np.zeros(points.shape[1])
    
    for j in range(k):
        mask = assignments == j
        centroids.append(
            (points[mask].mean(axis=0) if mask.any() else fallback).tolist()
        )

    return centroids
    