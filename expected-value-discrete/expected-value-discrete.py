import numpy as np
import math

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x_np, p_np = np.asarray(x), np.asarray(p)

    if not math.isclose(np.sum(p_np), 1.0, abs_tol=1e-6):
        raise ValueError("The sum of provided probabilities should be 1")

    if x_np.shape != p_np.shape:
        raise ValueError(f"The shape is mismatched, got x:{x_np.shape} and p:{p_np.shape}")
    
    return np.sum(np.asarray(x) * np.asarray(p))
