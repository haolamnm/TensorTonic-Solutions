import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positions = np.arange(seq_length).reshape(-1, 1) # (L, 1)
    dims = np.arange(0, d_model, 2) # (D/2,)

    angles = positions / (10_000**(2 * dims / d_model)) # (L, D/2)
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    
    return pe