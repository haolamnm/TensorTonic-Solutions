import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    B, N, D = patches.shape
    if cls_token is None:
        cls_token = np.random.randn(1, 1, D) * 0.02

    # (1, 1, D) -> (B, 1, D)
    # elegant np.tile
    cls_broadcast = np.tile(cls_token, (B, 1, 1))

    z_0 = np.concatenate([cls_broadcast, patches], axis=1)
    return z_0