import numpy as np

def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    values = np.asarray(values, dtype=float)
    ranks = np.argsort(values)
    sorts = values[ranks]

    groups = {}
    for rank in ranks:
        # 1-based rank
        groups.setdefault(sorts[rank], []).append(rank + 1)

    for value, ranks in groups.items():
        groups[value] = float(np.mean(ranks))

    result = []
    for value in values:
        result.append(groups[value])

    return result
    
    