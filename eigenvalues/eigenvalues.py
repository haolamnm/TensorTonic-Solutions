import numpy as np
import numpy.linalg as la

def is_square(matrix) -> bool:
    if not matrix or not isinstance(matrix[0], list):
        return False
    return len(matrix) == len(matrix[0])

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    if not is_square(matrix):
        return None
        
    matrix = np.asarray(matrix)
    
    eigenvalues, eigenvectors = la.eig(matrix)
    return eigenvalues