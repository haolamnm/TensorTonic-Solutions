import numpy as np


def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    ratings = np.asarray(ratings, dtype=int)
    similarities = np.asarray(similarities, dtype=float)

    excluded = np.maximum(0.0, similarities)
    weighted = np.sum(np.dot(excluded, ratings))
    total = np.sum(excluded)

    return weighted / total if total != 0.0 else 0.0