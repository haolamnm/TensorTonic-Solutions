import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # We have N points and K centroids.
    # We need to construct matrix D shape N x K.

    points = np.asarray(points) # (N, 2)
    centroids = np.asarray(centroids) # (K, 2)

    sq_points = np.sum(np.square(points), axis=1)
    sq_points = np.expand_dims(sq_points, axis=1) # (N, 1)
    
    sq_centroids = np.sum(np.square(centroids), axis=1)
    sq_centroids = np.expand_dims(sq_centroids, axis=0) # (1, K)

    # Distance from point i to centroid j
    dist = sq_points + sq_centroids - 2 * points @ centroids.T

    n = len(points)
    return [int(np.argmin(dist[i, :])) for i in range(n)]
