import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    """
    Returns: np.ndarray of the final denoised sample
    """
    x_t = np.array(x_T, dtype=np.float64)
    betas = np.array(betas)
    T = len(betas)
    
    # calculate alpha_bar with 6-decimal precision
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    alpha_bars = np.array([round(float(v), 6) for v in alpha_bars])

    # iterative denoising: T down to 1
    for i in range(T):
        t = T - i  # 3, 2, 1
        idx = t - 1
        
        alpha_t = alphas[idx]
        a_bar_t = alpha_bars[idx]
        beta_t = betas[idx]
        
        eps_pred = np.array(epsilon_preds[i])
        
        # posterior mean calculation
        coeff = 1 / np.sqrt(alpha_t)
        noise_coeff = beta_t / np.sqrt(1 - a_bar_t)
        mu = coeff * (x_t - noise_coeff * eps_pred)
        
        if t > 1:
            # add stochastic noise sigma_t * z
            z = np.array(z_values[i])
            sigma_t = np.sqrt(beta_t)
            x_t = mu + sigma_t * z
        else:
            # final step is deterministic
            x_t = mu
            
    return np.round(x_t, 4)