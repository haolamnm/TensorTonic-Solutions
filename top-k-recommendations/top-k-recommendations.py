import numpy as np

def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    scores = np.asarray(scores, dtype=float)
    indices = np.argsort(scores).tolist()
    print(indices)
    excluded_indices = set(rated_indices)

    view = indices if (scores == np.mean(scores)).all() else indices[::-1]

    recommendations = []
    for idx in view:
        if idx not in excluded_indices:
            recommendations.append(idx)
        if len(recommendations) == k:
            break

    return recommendations
