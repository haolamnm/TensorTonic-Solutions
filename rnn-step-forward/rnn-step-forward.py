import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    h_prev = np.asarray(h_prev)
    x_t = np.asarray(x_t)
    Wx = np.asarray(Wx)
    Wh = np.asarray(Wh)
    b = np.asarray(b)
    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b)
    return h_t