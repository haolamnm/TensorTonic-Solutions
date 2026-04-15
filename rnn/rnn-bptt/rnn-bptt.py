import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    dtanh = (1 - h_t**2) * dh_next
    dh_prev = dtanh @ W_hh
    dW_hh = dtanh.T @ h_prev

    return dh_prev, dW_hh