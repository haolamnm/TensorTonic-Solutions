import numpy as np

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    R = np.asarray(ratings_matrix, dtype=float)

    # pre-calc mean
    row_sums = R.sum(axis=1)
    row_counts = (R != 0).sum(axis=1)
    
    # avoid division by zero for users with no ratings
    user_means = np.divide(row_sums, row_counts, 
                           out=np.zeros_like(row_sums), 
                           where=row_counts != 0)
    
    # identify users who rated both item_i and item_j
    slice_i = R[:, item_i]
    slice_j = R[:, item_j]
    common_users_mask = (slice_i != 0) & (slice_j != 0)
    
    if not np.any(common_users_mask):
        return 0.0

    # extract ratings and means for common users
    r_ui = slice_i[common_users_mask]
    r_uj = slice_j[common_users_mask]
    r_u_mean = user_means[common_users_mask]

    # compute centered ratings
    centered_i = r_ui - r_u_mean
    centered_j = r_uj - r_u_mean

    # 6. calculate similarity
    num = np.sum(centered_i * centered_j)
    den = np.sqrt(np.sum(centered_i**2)) * np.sqrt(np.sum(centered_j**2))

    if den == 0:
        return 0.0
        
    return num / den