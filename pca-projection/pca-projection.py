import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X)
    n, d = X.shape

    # center
    Xc = X - X.mean(axis=0)

    # sample cov
    C = Xc.T @ Xc / (n - 1)

    # rng-based
    rng = np.random.default_rng(42)

    # power iteration with deflation
    W = []
    for _ in range(k):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)

        for _ in range(1_000):
            v_new = C @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break # zero eigenvalue, stop early
            v_new /= norm
            if np.abs(np.abs(v_new @ v) - 1.0) < 1e-10:
                v = v_new
                break
            v = v_new

        lam = v @ C @ v
        if lam < 1e-10:
            # remains are zero -> project to zeros
            W.append(np.zeros(d))
        else:
            W.append(v)
            C = C - lam * np.outer(v, v)

    W = np.column_stack(W) # d x k
    return (Xc @ W).tolist()

