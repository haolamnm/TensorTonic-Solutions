import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # N, H, W, C = x.shape
    
    top_left     = x[:, 0::2, 0::2, :]
    top_right    = x[:, 0::2, 1::2, :]
    bottom_left  = x[:, 1::2, 0::2, :]
    bottom_right = x[:, 1::2, 1::2, :]

    out = np.maximum(
        np.maximum(top_left, top_right),
        np.maximum(bottom_left, bottom_right)
    )
    return out