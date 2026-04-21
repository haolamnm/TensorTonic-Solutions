import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels.
    """
    B, H, W, C = x.shape

    sq = x**2

    pad_size = n // 2
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (pad_size, pad_size)
    padded_sq = np.pad(sq, pad_width, mode="constant")

    sum_sq = np.zeros_like(x)
    for i in range(n):
        slc = [slice(None)] * x.ndim
        slc[-1] = slice(i, i + C)
        sum_sq += padded_sq[tuple(slc)]

    denom = (k + alpha * sum_sq) ** beta
    return x / denom