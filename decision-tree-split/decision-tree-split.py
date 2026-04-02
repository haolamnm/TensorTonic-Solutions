import numpy as np


def calculate_gini(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs**2)

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    parent_gini = calculate_gini(y)
    best_gain = -float('inf')
    best_feature = -1
    best_threshold = -1.0

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_vals = np.sort(np.unique(feature_values))

        if len(unique_vals) < 2:
            continue

        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0

            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            weight_left = len(y_left) / n_samples
            weight_right = len(y_right) / n_samples

            gini_split = (weight_left * calculate_gini(y_left)) + (weight_right * calculate_gini(y_right))

            information_gain = parent_gini - gini_split

            if information_gain > best_gain:
                best_gain = information_gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold