import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # These are not allowed to use
    # return np.trace(A)
    # return np.sum(np.diag(A))
    
    if len(A) == 0:
        return 0.0
    if len(A[0]) == 0:
        return 0.0

    trace = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == j:
                trace += A[i][j]

    return trace
            
