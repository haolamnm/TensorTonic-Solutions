import numpy as np

def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    R = np.asarray(ratings_matrix, dtype=float)
    masked = np.where(R == 0, np.nan, R)

    mu = np.nanmean(masked)
    r_u = np.nanmean(masked, axis=1, keepdims=True) # (C, 1)
    r_i = np.nanmean(masked, axis=0, keepdims=True) # (1, I)

    b_u = np.where(np.isnan(r_u), 0, r_u - mu)
    b_i = np.where(np.isnan(r_i), 0, r_i - mu)

    pred_matrix = mu + b_u + b_i # (C, I) via broadcasting

    return [pred_matrix[u, i] for u, i in target_pairs]