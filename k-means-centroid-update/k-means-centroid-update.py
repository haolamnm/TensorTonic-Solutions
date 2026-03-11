import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.asarray(points)
    assignments = np.asarray(assignments)
    centroids = []
    for j in range(k):
        mask = assignments == j
        points_j = points[mask]
        print(points_j.shape)
        centroids.append(np.mean(points_j, axis=0).tolist())
        print(centroids)

    return centroids
    