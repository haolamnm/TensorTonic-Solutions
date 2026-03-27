import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=int)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=int)

    if len(y_train) == 0:
        return None

    # priors (C,)
    priors = np.bincount(y_train) / len(y_train)

    # mask (C, N)
    mask = (y_train == np.unique(y_train)[:, None]).astype(int)

    # X_train (N, D)
    # feature_counts (C, D)
    feature_counts = mask @ X_train
    print(feature_counts)

    # n_y (C,)
    n_y = mask.sum(axis=1)
    print(n_y)

    thetas = (feature_counts + 1) / (n_y[:, None] + 2)
    print(thetas)

    # bernoulli likelihood into linear
    # log(P) = x.log(thetas) + (1-x).log(1-thetas)
    # log(P) = x.(log(thetas) - log(1-thetas)) + log(1-thetas)
    # W = log(thetas) - log(1-thetas)
    # b = log(1-thetas)
    # y = X_test @ W + b
    W = np.log(thetas) - np.log(1 - thetas)
    b = np.log(priors) + np.sum(np.log(1 - thetas), axis=1)
    y = X_test @ W.T + b

    return y