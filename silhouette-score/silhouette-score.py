import numpy as np


def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    labels = np.asarray(labels)
    unique, inv = np.unique(labels, return_inverse=True)
    K = len(unique)
    N = len(labels)

    # pairwise euclidean distance (N, N)
    sq = (X ** 2).sum(axis=1)
    dist = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0, None))

    # one-hot cluster membership (N, K)
    # membership[i, j] means point_i in cluster_j
    membership = (inv[:, None] == np.arange(K)[None, :]).astype(float)
    cluster_sizes = membership.sum(axis=0)  # (K,)

    # sum of distances from each point to all points in each cluster (N, K)
    sum_dist = dist @ membership  # (N, K)

    # intra: avg dist to same-cluster points (exclude self, hence -1)
    same_cluster_size = cluster_sizes[inv] - 1                    # (N,)
    intra = sum_dist[np.arange(N), inv] / np.maximum(same_cluster_size, 1)

    # inter: avg dist to each other cluster (N, K), mask own cluster with inf
    avg_dist = sum_dist / np.maximum(cluster_sizes, 1)            # (N, K)
    avg_dist[membership.astype(bool)] = np.inf                    # mask own cluster
    inter = avg_dist.min(axis=1)                                  # (N,)

    # silhouette per point, mean over all
    s = (inter - intra) / np.maximum(inter, intra)
    return float(np.mean(s))
