from numpy.lib.stride_tricks import sliding_window_view


def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # clever numpy utils
    windows = sliding_window_view(X, window_shape=(pool_size, pool_size))
    strided_windows = windows[::stride, ::stride]
    result = strided_windows.max(axis=(2, 3))

    return result.tolist()