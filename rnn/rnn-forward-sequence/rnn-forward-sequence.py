import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size, T, input_dim = X.shape
    h_prev = h_0
 
    H = []
    for t in range(T):
        x = X[:, t, :]
        h_t = np.tanh(x @ W_xh.T + h_prev @ W_hh.T + b_h)
        H.append(h_t)
        h_prev = h_t

    # clean usage of np.stack
    H = np.stack(H, axis=1)
    return (H, H[:, -1, :])