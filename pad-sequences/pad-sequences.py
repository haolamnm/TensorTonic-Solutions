import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    max_len = max_len or max(len(seq) for seq in seqs)

    for idx, seq in enumerate(seqs):
        n = len(seq)
        if max_len > n:
            seq.extend([pad_value] * (max_len - n))
        else:
            seqs[idx] = seq[:max_len]

    return seqs