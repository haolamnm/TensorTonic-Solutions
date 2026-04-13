import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    out = x.copy()
    j = 0
    for c in config:
        if isinstance(c, int):
            W = np.asarray(conv_weights[j])
            b = np.asarray(conv_biases[j])
            out = out @ W + b
            out = np.maximum(0, out)
            j += 1

        elif isinstance(c, str) and c.upper() == 'M':
            out = maxpool_2x2(out)

    return out