import numpy as np


def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    targets = np.asarray(targets)
    categories = np.asarray(categories)

    uniques, inverse = np.unique(categories, return_inverse=True)
    means = np.bincount(inverse, weights=targets) / np.bincount(inverse)

    return means[inverse].tolist()

