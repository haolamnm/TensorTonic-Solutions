import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate and hidden state."""
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)

    input = np.concatenate([h_prev, x_t], axis=-1)
    o_t = sigmoid((W_o @ input.T).T + b_o)
    h_t = o_t * np.tanh(C_t)

    o_t = o_t.squeeze(0) if o_t.shape[0] == 1 else o_t
    h_t = h_t.squeeze(0) if h_t.shape[0] == 1 else h_t

    return (o_t, h_t)