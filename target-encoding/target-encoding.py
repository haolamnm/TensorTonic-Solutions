import numpy as np


def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    targets = np.asarray(targets)
    categories = np.asarray(categories)
    uniques = np.unique(categories)

    category2encoding = {}
    for category in uniques:
        # retrieve scores within a category
        masked = targets[categories == category]
        category2encoding[category] = np.mean(masked)
        
    encoding = np.zeros(categories.shape)

    for index, category in enumerate(categories):
        encoding[index] = category2encoding[category]

    return encoding.tolist()