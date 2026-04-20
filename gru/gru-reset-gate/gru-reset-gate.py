import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    input = np.concatenate([h_prev, x_t], axis=-1)
    r_t = sigmoid((W_r @ input.T).T + b_r)
    return r_t.squeeze(0) if r_t.shape[0] == 1 else r_t