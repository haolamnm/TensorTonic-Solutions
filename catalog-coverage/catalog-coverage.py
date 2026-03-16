from itertools import chain

def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    n_uniq = len(set(chain.from_iterable(recommendations)))
    return n_uniq / n_items