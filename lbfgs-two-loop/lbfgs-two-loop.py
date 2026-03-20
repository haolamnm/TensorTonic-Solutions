import numpy as np

def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # phase 0: setup
    G = np.asarray(grad, dtype=float)
    S = np.asarray(s_list, dtype=float)
    Y = np.asarray(y_list, dtype=float)

    Q = np.copy(grad)
    
    P = [] # list of rho values
    for s, y in zip(S, Y):
        P.append(1.0 / _dot(s, y))

    # phase 1:
    A = [] # list of alpha values
    for s, y, p in zip(S[::-1], Y[::-1], P[::-1]):
        print(s, y, p)
        a = p * _dot(s, Q)
        Q = Q - a * y
        A.append(a)
    
    # phase 2:
    s_last = S[-1]
    y_last = Y[-1]
    
    gamma = _dot(s_last, y_last) / _dot(y_last, y_last)
    R = gamma * Q
    
    # phase 3:
    for s, y, p, a in zip(S, Y, P, A[::-1]):
        b = p * _dot(y, R)
        R = R + s * (a - b)
    
    return -R