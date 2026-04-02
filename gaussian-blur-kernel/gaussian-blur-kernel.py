import math as m

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    assert size % 2 == 1, "Size must be odd"
    center = size // 2
    G = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            y = i - center
            x = j - center

            G[i][j] = m.e ** (-(x**2 + y**2) / (2 * sigma**2))

    S = sum(sum(row) for row in G)
    for i in range(size):
        for j in range(size):
            G[i][j] /= S

    return G