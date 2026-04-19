import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    var = np.exp(log_var)
    D_KL = -0.5 * np.sum(1 + log_var - mu**2 - var, axis=1)
    R_KL = np.mean(D_KL, axis=0)

    MSE = np.sum((x - x_recon)**2, axis=1).mean()
    loss = MSE + R_KL

    return {
        "kl": R_KL,
        "recon": MSE,
        "total": loss
    }
