import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    c_t = np.concatenate([h_prev, x_t], axis=1) # (N, H + D)
    f_t = sigmoid((W_f @ c_t.T).T + b_f)
    return f_t.squeeze(0) if f_t.shape[0] == 1 else f_t