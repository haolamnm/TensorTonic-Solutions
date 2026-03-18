import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    xt = np.atleast_2d(x)
    h_prev = np.atleast_2d(h_prev)
    
    Wz = np.asarray(params["Wz"])
    Wr = np.asarray(params["Wr"])
    Wh = np.asarray(params["Wh"])

    Uz = np.asarray(params["Uz"])
    Ur = np.asarray(params["Ur"])
    Uh = np.asarray(params["Uh"])

    bz = np.asarray(params["bz"])
    br = np.asarray(params["br"])
    bh = np.asarray(params["bh"])

    # xt (N, D)
    # Wz (D, H)
    # Uz (H, H)
    # h_prev (N, H)
    # bz (H,)
    # (N, H) + (N, H) + (H,) = (N, H)
    zt = _sigmoid(xt @ Wz + h_prev @ Uz + bz)

    # (N, H) + (N, H) + (H,) = (N, H)
    rt = _sigmoid(xt @ Wr + h_prev @ Ur + br)

    # Wh (D, H)
    # (N, D) @ (D, H) = (N, H) + (N, H)
    h_hidden = np.tanh(xt @ Wh + (rt * h_prev) @ Uh + bh)
    ht = ((1 - zt) * h_prev) + (zt * h_hidden)

    return ht.squeeze(0) if len(ht) == 1 else ht