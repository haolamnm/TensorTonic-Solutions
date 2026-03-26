import numpy as np

def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    user_ratings = np.asarray(user_ratings, dtype=int)
    item_similarities = np.asarray(item_similarities, dtype=float)

    included_indices = (user_ratings != 0.0) & (item_similarities > 0)
    included_indices[target] = False
    
    included_ratings = user_ratings[included_indices]
    included_sims = item_similarities[included_indices]

    weighted = np.sum(np.dot(included_ratings, included_sims))
    total = np.sum(included_sims)

    return weighted / total if total != 0.0 else 0.0