import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    E = np.mean(x, axis=-1, keepdims=True)
    V = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - E) / np.sqrt(V + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    Q_full = Q @ W_q
    K_full = K @ W_k
    V_full = V @ W_v

    B, N, D_model = Q_full.shape
    _, M, _       = K_full.shape
    H = num_heads
    D_k = D_v = D_model // H

    Q_reshaped = Q_full.reshape(B, N, H, D_k).transpose(0, 2, 1, 3)
    K_reshaped = K_full.reshape(B, M, H, D_k).transpose(0, 2, 1, 3)
    V_reshaped = V_full.reshape(B, M, H, D_v).transpose(0, 2, 1, 3)

    S = (Q_reshaped @ K_reshaped.swapaxes(-1, -2)) / np.sqrt(D_k)
    A = softmax(S, axis=3) @ V_reshaped
    A_reshaped = A.reshape(B, M, D_model)
    
    return A_reshaped @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # A (B, L, D_model)
    # x (B, L, D_model)
    A = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_new = layer_norm(x + A, gamma1, beta1)
    F = feed_forward(x_new, W1, b1, W2, b2)
    out = layer_norm(x_new + F, gamma2, beta2)
    return out