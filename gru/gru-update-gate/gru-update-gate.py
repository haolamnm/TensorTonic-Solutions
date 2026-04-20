import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """
    Compute update gate: z_t = sigmoid(W_z @ [h, x] + b_z)
    """
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)

    input = np.concatenate([h_prev, x_t], axis=-1)
    z_t = sigmoid((W_z @ input.T).T + b_z)
    z_t = z_t.squeeze(0) if z_t.shape[0] == 1 else z_t

    return z_t