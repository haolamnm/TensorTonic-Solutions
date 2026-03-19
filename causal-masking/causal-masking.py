import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.asarray(scores, dtype=float)
    *_, h, w = scores.shape

    # debug experience: do not use int type for masking, cause
    # NumPy will interpret it as integer indexing
    mask = np.triu(np.ones((h, w), dtype=bool), k=1)  # (T, T)

    masked_scores = np.copy(scores)
    masked_scores[..., mask] = mask_value
    return masked_scores
