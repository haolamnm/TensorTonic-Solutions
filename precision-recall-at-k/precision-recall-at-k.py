def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k_set = set(recommended[:k])
    relevant_set = set(relevant)

    if not relevant_set:
        return [0.0, 0.0]
    
    hits = top_k_set & relevant_set
    precision = len(hits) / k
    recall = len(hits) / len(relevant_set)
    
    return [precision, recall]