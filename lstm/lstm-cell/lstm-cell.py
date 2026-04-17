import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    was_1d = x_t.ndim == 1
    
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    input = np.concatenate([h_prev, x_t], axis=-1)

    f_t = sigmoid((W_f @ input.T).T + b_f)

    i_t = sigmoid((W_i @ input.T).T + b_i)
    C_t_tilde = np.tanh((W_c @ input.T).T + b_c)

    o_t = sigmoid((W_o @ input.T).T + b_o)
    C_t = f_t * C_prev + i_t * C_t_tilde

    h_t = o_t * np.tanh(C_t)

    if was_1d:
        h_t = h_t.squeeze(0)
        C_t = C_t.squeeze(0)

    return (h_t, C_t)