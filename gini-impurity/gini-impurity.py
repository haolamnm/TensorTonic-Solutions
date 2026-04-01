import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_l_uniq, y_l_cnts = np.unique(y_left, return_counts=True)
    n_l = len(y_left)
    
    y_r_uniq, y_r_cnts = np.unique(y_right, return_counts=True)
    n_r = len(y_right)

    n = n_l + n_r
    if n == 0:
        return 0.0

    gini_l = 1 - np.sum((y_l_cnts / n_l)**2) if n_l else 0.0
    gini_r = 1 - np.sum((y_r_cnts / n_r)**2) if n_r else 0.0
    # print(gini_l)
    # print(gini_r)

    gini_split = (n_r / n) * gini_r + (n_l / n) * gini_l
    return gini_split