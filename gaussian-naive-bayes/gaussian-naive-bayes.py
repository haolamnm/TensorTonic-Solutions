import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test, dtype=float)

    classes = np.unique(y_train)
    n_total = len(y_train)

    log_priors = []
    means = []
    variances = []


    for c in classes:
        X_c = X_train[y_train == c]
        log_priors.append(np.log(len(X_c) / n_total))
        means.append(X_c.mean(axis=0))
        variances.append(X_c.var(axis=0) + 1e-9)

    log_priors = np.array(log_priors)          # (C,)
    means      = np.array(means)               # (C, F)
    variances  = np.array(variances)           # (C, F)

    # (N, C, F)
    diff = X_test[:, np.newaxis, :] - means

    log_likelihood = (
        -0.5 * np.log(2 * np.pi * variances)
        - (diff ** 2) / (2 * variances)
    ).sum(axis=2)                              # (N, C)

    log_posterior = log_priors + log_likelihood  # (N, C)

    return classes[np.argmax(log_posterior, axis=1)].tolist()