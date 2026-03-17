import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    batch, n, d_k = Q.size()

    # Q (B, N, D_K)
    # K (B, M, D_K) -> (B, D_K, M)
    S = Q @ K.swapaxes(-1, -2) # (B, N, M)
    S_scaled = S / math.sqrt(d_k)
    
    # (B, N, M)
    weights = F.softmax(S_scaled, dim=2)

    # V (B, M, D_V)
    return weights @ V