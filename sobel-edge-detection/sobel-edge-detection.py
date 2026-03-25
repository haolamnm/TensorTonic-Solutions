import numpy as np

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    image = np.array(image, dtype=float)
    padded = np.pad(image, 1, mode='constant', constant_values=0)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)

    H, W = image.shape
    Gx = np.zeros((H, W), dtype=float)
    Gy = np.zeros((H, W), dtype=float)

    # Extract sliding windows via stride tricks for vectorized convolution
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

    Gx = (windows * Kx).sum(axis=(-2, -1))
    Gy = (windows * Ky).sum(axis=(-2, -1))

    return np.sqrt(Gx ** 2 + Gy ** 2).tolist()
