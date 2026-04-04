import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X, y = np.array(X), np.array(y)

    train_idx, test_idx =[], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        # print(cls_idx)

        if rng is not None:
            rng.shuffle(cls_idx)
        else:
            np.random.shuffle(cls_idx)

        # np.floor = builtin int
        n_test = max(1, int(len(cls_idx) * test_size))

        if n_test >= len(cls_idx):
            n_test = len(cls_idx) - 1

        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])

    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.sort(np.concatenate(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
