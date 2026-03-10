import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w_np = np.asarray(w, dtype=float)
    g_np = np.asarray(g, dtype=float)
    s_np = np.asarray(s, dtype=float)
    
    s_new = beta * s_np + (1 - beta) * np.square(g_np)
    w_new = w_np - lr / np.sqrt(s_new + eps) * g_np
    
    return (w_new, s_new)