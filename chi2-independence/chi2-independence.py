import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    C = np.asarray(C, dtype=float)
    
    row_sums = C.sum(axis=1, keepdims=True)  # (r, 1)
    col_sums = C.sum(axis=0, keepdims=True)  # (1, c)
    total = C.sum()
    
    E = (row_sums * col_sums) / total # outer product / total
    
    chi2 = np.sum((C - E) ** 2 / E)
    
    return float(chi2), E