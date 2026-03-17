import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Q = (B, N, D_model)
    # K = (B, M, D_model)
    # V = (B, M, D_model)
    # W_q = W_k = W_v = (D_model, D_model)
    # Q_full = (B, N, D_model)
    # K_full = (B, M, D_model)
    # V_full = (B, M, D_model)
    Q_full = Q @ W_q
    K_full = K @ W_q
    V_full = V @ W_v


    B, N, D_model = Q_full.shape
    _, M, _       = K_full.shape
    H = num_heads
    D_v = D_k = D_model // H

    Q_reshaped = Q_full.reshape(B, N, H, D_k).transpose(0, 2, 1, 3)
    K_reshaped = K_full.reshape(B, M, H, D_k).transpose(0, 2, 1, 3)
    V_reshaped = V_full.reshape(B, M, H, D_v).transpose(0, 2, 1, 3)

    # (B, H, N, M)
    scores = (Q_reshaped @ K_reshaped.swapaxes(-1, -2)) / np.sqrt(D_k)
    
    # Attention, (B, H, N, M) @ (B, H, M, D_v)
    A = softmax(scores, axis=3) @ V_reshaped
    A_reshaped = A.reshape(B, M, D_model)

    return A_reshaped @ W_o