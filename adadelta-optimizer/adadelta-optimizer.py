import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq)
    E_update_sq = np.asarray(E_update_sq)
    
    E_grad_sq_new = rho * E_grad_sq + (1 - rho) * grad**2
    w_delta = -(np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq_new + eps)) * grad
    E_update_sq_new = rho * E_update_sq + (1 - rho) * w_delta**2
    w_new = w + w_delta

    return w_new, E_grad_sq_new, E_update_sq_new