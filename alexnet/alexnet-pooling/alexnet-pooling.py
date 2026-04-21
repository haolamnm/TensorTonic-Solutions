import numpy as np
import math as m

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Apply 2D max pooling (shape simulation).
    """
    B, H, _, C = x.shape
    H_out = m.floor((H - kernel_size) / stride) + 1
    out = np.zeros((B, H_out, H_out, C))

    return out