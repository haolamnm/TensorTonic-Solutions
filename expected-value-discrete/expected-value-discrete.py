import numpy as np
import math

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x_np, p_np = np.asarray(x), np.asarray(p)

    if x_np.shape != p_np.shape:
        raise ValueError(f"Shape mismatch: x={x_np.shape}, p={p_np.shape}")
        
    if not math.isclose(np.sum(p_np), 1.0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to 1")
    
    return np.dot(x_np, p_np)
