import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    y = np.zeros((N, C_out, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, i:i+KH, j:j+KW] # (N, C_in, KH, KW)
            # clean use of einsum
            y[:, :, i, j] = np.einsum('nchw,ochw->no', patch, W) + b

    return y