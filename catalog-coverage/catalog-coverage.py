def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    chained_recommendations = []
    for rec in recommendations:
        chained_recommendations.extend(rec)
    n_uniq = len(set(chained_recommendations))
    return n_uniq / n_items