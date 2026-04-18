import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape
    P = patch_size
    N = (H // P) * (W // P)
    D = P * P * C

    if W_proj is None:
        W_proj = np.random.randn(D, embed_dim)

    # (B, H, W, C) -> (B, H // P, P, W // P, P, C)
    patches = image.reshape(B, H // P, P, W // P, P, C)

    # (B, H // P, P, W // P, P, C)
    # -> (B, H_new, W_new, P, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)

    patches = patches.reshape(B, N, D)
    
    embeddings = patches @ W_proj
    return embeddings