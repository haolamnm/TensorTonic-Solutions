import numpy as np

def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    scores = np.asarray(scores, dtype=float)
    excluded = set(rated_indices)
    
    sorted_indices = np.argsort(-scores).tolist()
    recommendations = [idx for idx in sorted_indices if idx not in excluded]
    # might use early break if K is large
    
    return recommendations[:k]
