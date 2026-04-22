import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    x_t = np.array(x_t)
    eps_pred = np.array(epsilon_pred)
    betas = np.array(betas)

    idx = t - 1
    beta_t = betas[idx]
    alpha_t = 1 - beta_t

    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    alpha_bar_t = round(float(alpha_bars[idx]), 6)

    coeff = 1 / np.sqrt(alpha_t)
    noise_coeff = beta_t / np.sqrt(1 - alpha_bar_t)
    mu = coeff * (x_t - noise_coeff * eps_pred)

    if t == 1:
        # step t=1 is deterministic (z=0)
        return np.round(mu, 4)

    # stochastic part for t > 1: add sigma_t * z
    sigma_t = np.sqrt(beta_t)
    if z is None:
        z = np.random.normal(size=x_t.shape)
    else:
        z = np.array(z)

    x_prev = mu + sigma_t * z
    return np.round(x_prev, 4)