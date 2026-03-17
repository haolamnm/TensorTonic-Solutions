import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    if max_norm <= 0:
        return np.asarray(g)
    
    g = np.atleast_2d(g)
    g_norm = np.linalg.norm(g)

    scale = max_norm / g_norm
    g_clipped = np.where(g_norm <= max_norm, g, g * scale)
    result = g_clipped.squeeze(0) if len(g_clipped) == 1 else g_clipped
    
    return result