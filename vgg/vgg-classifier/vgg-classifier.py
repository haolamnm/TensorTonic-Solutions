import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    B, H, W, C = features.shape
    flat = features.reshape(B, -1)

    fc1 = flat @ W1 + b1
    h1 = np.maximum(0, fc1)

    fc2 = h1 @ W2 + b2
    h2 = np.maximum(0, fc2)

    fc3 = h2 @ W3 + b3
    return fc3