import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)
    """
    r_t = np.atleast_2d(r_t)
    x_t = np.atleast_2d(x_t)
    h_prev = np.atleast_2d(h_prev)
    
    input = np.concatenate([r_t * h_prev, x_t], axis=-1)
    h_new_t = np.tanh((W_h @ input.T).T + b_h)
    h_new_t = h_new_t.squeeze(0) if h_new_t.shape[0] == 1 else h_new_t

    return h_new_t