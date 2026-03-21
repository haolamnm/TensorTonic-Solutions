import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    if x.ndim == 4:
        axis = (0, 2, 3)
        _, C, _, _ = x.shape
        gamma = gamma.reshape(1, C, 1, 1)
        beta = beta.reshape(1, C, 1, 1)
        
    elif x.ndim == 2:
        axis = 0
        
    else:
        return None
    
    
    nuy = np.mean(x, axis=axis, keepdims=True)
    sigma_sq = np.var(x, axis=axis, keepdims=True)

    x_hat = (x - nuy) / np.sqrt(sigma_sq + eps)
    y = gamma * x_hat + beta

    return y