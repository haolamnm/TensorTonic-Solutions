import numpy as np


def cluster_distances(dist, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    N = len(labels)

    intra = np.zeros(N)
    inter = np.zeros(N)

    for k in unique:
        mask_k = labels == k
        mask_other = ~mask_k

        # intra: avg dist to other points in same cluster
        same_count = mask_k.sum() - 1
        if same_count > 0:
            intra[mask_k] = dist[np.ix_(mask_k, mask_k)].sum(axis=1) / same_count

        # inter: min avg dist over all other clusters
        other_labels = unique[unique != k]
        inter_avgs = []
        for j in other_labels:
            mask_j = labels == j
            avg_dist_to_j = dist[np.ix_(mask_k, mask_j)].mean(axis=1)
            inter_avgs.append(avg_dist_to_j)

        inter[mask_k] = np.min(inter_avgs, axis=0)

    return intra, inter


def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)

    sq = (X ** 2).sum(axis=1)
    dist = np.sqrt(sq[:, None] + sq[None, :] - 2 * X @ X.T)
    dist = np.clip(dist, 0, None)

    intra, inter = cluster_distances(dist, labels)

    scores = (inter - intra) / np.maximum(intra, inter)

    return scores.mean()
