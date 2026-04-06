import numpy as np

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    ratings_matrix = np.asarray(ratings_matrix)
    slice_i = ratings_matrix[:, item_i]
    slice_j = ratings_matrix[:, item_j]
    print(slice_i)
    print(slice_j)

    mask = (slice_i != 0) & (slice_j != 0)
    mask = np.broadcast_to(mask, ratings_matrix.T.shape)
    mask = mask.T
    print(mask)

    users = ratings_matrix[mask].reshape(-1, ratings_matrix.shape[1])
    print(users)
    
    if not mask.any():
        return 0.0

    nums = []
    denos_a = []
    denos_b = []
    for user in users.tolist():
        r_ui = user[item_i]
        r_uj = user[item_j]
        u = np.array(user)
        r_u = u[u != 0].mean()
        nums.append((r_ui - r_u) * (r_uj - r_u))
        denos_a.append((r_ui - r_u)**2)
        denos_b.append((r_uj - r_u)**2)

    return np.sum(nums) / (np.sqrt(np.sum(denos_a)) * np.sqrt(np.sum(denos_b)))