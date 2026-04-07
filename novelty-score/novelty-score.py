import numpy as np

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    R = np.asarray(recommendations)
    C = np.asarray(item_counts)
    N = n_users

    novelty = np.mean(-np.log2(C[R] / N))
    return float(novelty)