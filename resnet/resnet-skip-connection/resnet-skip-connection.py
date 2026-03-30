import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    grad_total = np.eye(gradients_F[0].shape[0])

    for grad in gradients_F:
        identity = np.eye(grad.shape[0])
        grad_total = (identity + grad) @ grad_total

    return (grad_total @ x.T).T

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    grad_total = np.eye(gradients_F[0].shape[0])
    
    for grad in gradients_F:
        grad_total = grad @ grad_total

    return (grad_total @ x.T).T
