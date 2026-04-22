import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    eps = np.array(epsilon)
    eps_p = np.array(epsilon_pred)

    loss = np.mean((eps - eps_p)**2)
    
    return float(loss)