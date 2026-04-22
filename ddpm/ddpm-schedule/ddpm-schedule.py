import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    betas = np.linspace(beta_1, beta_T, T)
    return [round(float(v), 6) for v in betas]

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    steps = np.arange(T + 1)

    # f(t) = cos(((t/T + s) / (1 + s)) * pi/2)^2
    f_t = np.cos(((steps / T + s) / (1 + s)) * (np.pi / 2))**2
    alpha_bars = f_t / f_t[0]

    # return T values: alpha_bar_1 to alpha_bar_T
    alpha_bars_t = alpha_bars[1:]
    clipped = np.clip(alpha_bars_t, 0.0001, 0.9999)
    
    return [round(float(v), 6) for v in clipped]

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    alpha_bars = np.array(alpha_bars)
    # alpha_bar_0 is 1.0 by definition
    alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
    
    # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1 - (alpha_bars / alpha_bars_prev)
    clipped = np.clip(betas, 0.0001, 0.9999)
    
    return [round(float(v), 6) for v in clipped]