import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    # X (B, N, D_model)
    # nuy (B, N, 1)
    # sigma (B, N, 1)
    E = np.mean(x, axis=-1, keepdims=True)
    V = np.var(x, axis=-1, keepdims=True)

    # gamma (D_model,)
    # beta (D_model,)
    norm = gamma * (x - E) / np.sqrt(V + eps) + beta
    return norm