import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p, q = np.asarray(p), np.asarray(q)

    # positive probs masking
    mask = p > 0
    p, q = p[mask], q[mask]
    
    dkl = np.sum(p * np.log(p / (q + eps)))

    return dkl