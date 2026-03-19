import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    x_t = np.asarray(cache[0], dtype=float)
    h_prev = np.asarray(cache[1], dtype=float)
    h_t = np.asarray(cache[2], dtype=float)
    W = np.asarray(cache[3], dtype=float)
    U = np.asarray(cache[4], dtype=float)
    b = np.asarray(cache[5], dtype=float)
    
    dh = np.asarray(dh, dtype=float)

    dh_t = 1 - h_t**2 # dh_t/dz
    dz = dh * dh_t # dL/dz = dL/dh_t * dh_t/dz

    dW = np.outer(dz, x_t) # dL/dW
    dU = np.outer(dz, h_prev) # dL/dU
    db = dz

    dx_t = W.T @ dz
    dh_prev = U.T @ dz

    return dx_t, dh_prev, dW, dU, db
