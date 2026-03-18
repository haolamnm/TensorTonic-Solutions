import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len).reshape(-1, 1) # (N,1)
    dims = np.arange((d_model + 1) // 2).reshape(1, -1) # (1, ceil(D/2))
    
    # (N, 1) / (1, D/2) = (N, ceil(D/2))
    angles = positions / np.power(base, (2 * dims / d_model))
    print(angles.shape)
    print(angles)

    # (N, D)
    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(angles)
    if d_model % 2 == 1:
        pe[:, 1::2] = np.cos(angles[:, :-1])
    else:
        pe[:, 1::2] = np.cos(angles)

    return pe