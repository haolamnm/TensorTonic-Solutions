import numpy as np
import math as m

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-8):
    x_mean = x.mean(axis=-1, keepdims=True)
    x_cen = x - x_mean

    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)

    x_hat = x_cen / std
    return x_hat

def gelu(x):
    out = 0.5 * x * (1 + np.tanh(np.sqrt(2 / m.pi) * (x + 0.044715 * x**3)))
    return out

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    B, N, D = x.shape
    h = num_heads
    d_k = D // h
    x_hat = layer_norm(x)
    
    Q_full, K_full, V_full = x_hat @ Wq, x_hat @ Wk, x_hat @ Wv
    
    # (B, N, D) -> (B, N, h, d_k) -> (B, h, N, d_k)
    def split_heads(t):
        return t.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    
    Q, K, V = split_heads(Q_full), split_heads(K_full), split_heads(V_full)
    
    # (B, h, N, d_k) @ (B, h, d_k, N) -> (B, h, N, N)
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    
    # (B, h, N, N) @ (B, h, N, d_k) -> (B, h, N, d_k)
    out_heads = attn @ V
    
    # (B, h, N, d_k) -> (B, N, h, d_k) -> (B, N, D)
    msa_out = out_heads.transpose(0, 2, 1, 3).reshape(B, N, D)
    
    x_prime = x + msa_out @ Wo
    
    x_prime_hat = layer_norm(x_prime)
    mlp_out = gelu(x_prime_hat @ W1) @ W2
    
    return x_prime + mlp_out