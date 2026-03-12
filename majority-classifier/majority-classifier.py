import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # clean usage of np.bincount
    counts = np.bincount(y_train)
    pred = np.argmax(counts)

    # clean usage of np.full
    return np.full(np.asarray(X_test).shape, pred).tolist()