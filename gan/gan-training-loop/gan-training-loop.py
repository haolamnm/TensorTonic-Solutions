import numpy as np
    
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def safe_clip(x, eps=1e-8):
    return np.clip(x, eps, 1-eps)

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)
    
    real_probs = sigmoid(real_data @ D_W)
    fake_probs = sigmoid(fake_data @ D_W)

    real_probs = safe_clip(real_probs)
    fake_probs = safe_clip(fake_probs)

    L_D = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    L_G = -np.mean(np.log(fake_probs))

    return {
        "d_loss": L_D,
        "g_loss": L_G
    }