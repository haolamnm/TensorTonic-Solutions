import numpy as np

def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    if not items:
        return []
        
    items_np = np.asarray(items)
    votes_np, cnts_np = items_np[:, 0], items_np[:, 1]

    # Weighted rating vectorized
    wr_np = (cnts_np * votes_np + min_votes * global_mean) / (cnts_np + min_votes)

    return wr_np.tolist()