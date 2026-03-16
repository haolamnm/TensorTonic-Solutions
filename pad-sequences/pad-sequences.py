import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    L = max_len or max(len(seq) for seq in seqs)
    N = len(seqs)

    results = np.full((N, L), fill_value=pad_value)
    for idx, seq in enumerate(seqs):
        n = min(len(seq), L)
        results[idx, :n] = seq[:n]
    
    return results