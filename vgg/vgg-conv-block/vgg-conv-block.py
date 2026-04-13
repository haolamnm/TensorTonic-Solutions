import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    out = x.copy()
    for w, b in zip(weights, biases, strict=True):
        w = np.asarray(w)
        b = np.asarray(b)
        out = out @ w + b
        out = np.maximum(0, out)
        
    # return np.maximum(0, out)
    return out