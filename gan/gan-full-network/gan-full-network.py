import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    
def safe_clip(x, eps=1e-8):
    return np.clip(x, eps, 1-eps)

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        z_np = np.asarray(z)
        samples = np.round(np.tanh(z_np @ self.G_W), 4)
        return samples.tolist()
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        x_np = np.asarray(x)
        samples = np.round(sigmoid(x_np @ self.D_W), 4)
        return samples.tolist()
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        fake_samples = np.asarray(self.generate(z))
        real_probs = safe_clip(np.asarray(self.discriminate(real_data)))
        fake_probs = safe_clip(np.asarray(self.discriminate(fake_samples)))
        
        L_D = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
        L_G = -np.mean(np.log(fake_probs))

        return {
            "d_loss": L_D,
            "g_loss": L_G
        }