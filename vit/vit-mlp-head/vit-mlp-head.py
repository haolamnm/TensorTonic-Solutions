import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    B, N, D = encoder_output.shape
    h_cls = encoder_output[:, 0, :]
    C = num_classes
    
    if W_head is None:
        W_head = np.random.randn(D, C) * 0.02
    
    eps = 1e-8
    h_mean = h_cls.mean(axis=1, keepdims=True)
    h_std = h_cls.std(axis=1, keepdims=True)
    h_hat = (h_cls - h_mean) / (h_std + eps)

    logits = h_hat @ W_head
    
    return logits
    