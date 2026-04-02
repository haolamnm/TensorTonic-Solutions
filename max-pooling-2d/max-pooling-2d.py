import numpy as np

def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    X = np.asarray(X)
    p = pool_size
    H, W = X.shape
    H_out, W_out = H // p, W // p

    out = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # slice the p x p window directly
            window = X[i*p : i*p+p, j*p : j*p+p]
            out[i, j] = np.max(window)
            
    return out.tolist()