import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    c_t = np.concatenate([h_prev, x_t], axis=1) # (N, H + D)
    i_t = sigmoid((W_i @ c_t.T).T + b_i)
    c_tilde = np.tanh((W_c @ c_t.T).T + b_c)

    i_t = i_t.squeeze(0) if i_t.shape[0] == 1 else i_t
    c_tilde = c_tilde.squeeze(0) if c_tilde.shape[0] == 1 else c_tilde

    return (i_t, c_tilde)