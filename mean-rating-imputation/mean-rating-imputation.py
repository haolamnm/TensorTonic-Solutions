import numpy as np

def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    mode2axis = {
        "user": 1,
        "item": 0
    }

    matrix = np.asarray(ratings_matrix, dtype=float)
    # replace with nans, to leverage np.nanmean
    mask = matrix == 0
    matrix_with_nans = np.where(mask, np.nan, matrix)

    with np.errstate(invalid='ignore'):
        means = np.nanmean(matrix_with_nans, axis=mode2axis.get(mode, -1), keepdims=True)

    means = np.nan_to_num(means, nan=0.0)
    broadcasted_means = np.broadcast_to(means, matrix.shape)

    matrix[mask] = broadcasted_means[mask]
    return matrix.tolist()