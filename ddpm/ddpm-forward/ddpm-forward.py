import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    alphas = 1 - np.array(betas)
    alpha_bar = np.cumprod(alphas)
    return [round(float(v), 6) for v in alpha_bar]

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    a_bars = get_alpha_bar(betas)
    a_bar_t = a_bars[t-1]

    x0_arr = np.array(x_0)
    eps_arr = np.array(epsilon)

    x_t = np.sqrt(a_bar_t) * x0_arr + np.sqrt(1 - a_bar_t) * eps_arr
    return np.round(x_t, 4).tolist()