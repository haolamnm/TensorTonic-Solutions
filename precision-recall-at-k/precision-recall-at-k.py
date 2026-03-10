def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:k]
    top_k_set = set(top_k)
    relevant_set = set(relevant)
    hits = top_k_set.intersection(relevant_set)

    precision = len(hits) / k
    recall = len(hits) / len(relevant_set)

    return [precision, recall]