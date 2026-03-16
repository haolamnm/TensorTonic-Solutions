import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    values = np.sort(np.asarray(x, dtype=float))
    percentiles = np.asarray(q, dtype=float) / 100

    indicies = (len(values) - 1) * percentiles
    floors = np.floor(indicies).astype(int)
    ceils = np.ceil(indicies).astype(int)
    weights = indicies - floors

    print(indicies)
    print(floors)
    print(ceils)
    print(weights)
    
    return weights * (values[ceils] - values[floors]) + values[floors]