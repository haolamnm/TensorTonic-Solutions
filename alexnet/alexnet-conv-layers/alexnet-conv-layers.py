import numpy as np
import math as m

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    B, H_in, _, C_in = image.shape

    k, s, p, f = 11, 4, 2, 96
    
    H_out = m.floor((H_in + 2*p - k) / s + 1)
    out = np.zeros((B, H_out, H_out, f))
    return out