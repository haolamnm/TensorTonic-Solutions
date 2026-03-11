def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    weighted_rating = []
    for vote, cnt in items:
        num = cnt * vote + min_votes * global_mean
        deno = cnt + min_votes
        weighted_rating.append(num / deno)

    return weighted_rating
    
    