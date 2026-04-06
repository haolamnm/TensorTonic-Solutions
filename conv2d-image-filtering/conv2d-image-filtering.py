import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    padded = np.pad(image, pad_width=padding, mode="constant", constant_values=0)
    # print(padded)

    windows = sliding_window_view(padded, window_shape=kernel.shape)
    # print(len(windows))

    strided = windows[::stride, ::stride]
    # print(len(strided))
    # print(strided.shape)
    # print(kernel.shape)

    h_out, w_out, _, _ = strided.shape
    out = []
    for i in range(h_out):
        line = []
        for j in range(w_out):
            val = int(np.sum(strided[i, j, :, :] * kernel))
            line.append(val)
        out.append(line)

    return out

