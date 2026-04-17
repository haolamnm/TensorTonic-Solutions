import numpy as np

def safe_clip(probs, eps=1e-8):
    return np.clip(probs, eps, 1-eps)

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    real_probs = safe_clip(real_probs)
    fake_probs = safe_clip(fake_probs)
    
    L_D = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    return L_D

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    fake_probs = safe_clip(fake_probs)
    
    L_G = -np.mean(np.log(fake_probs))
    return L_G