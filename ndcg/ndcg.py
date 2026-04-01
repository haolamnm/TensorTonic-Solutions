import math
import numpy as np

def dcg(relevance_scores, k) -> float:
    rel = np.asarray(relevance_scores, dtype=float)[:k]
    indices = np.arange(1, k + 1)

    dcg_at_k = np.sum((2 ** rel - 1) / np.log2(indices + 1))
    return float(dcg_at_k)

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    if k < 1:
        raise ValueError

    if k > len(relevance_scores):
        k = len(relevance_scores)
    
    rel = np.asarray(relevance_scores, dtype=float)
    rel_sorted = -np.sort(-rel)

    dcg_at_k = dcg(rel, k)
    idcg_at_k = dcg(rel_sorted, k)
    print(dcg_at_k)
    print(idcg_at_k)

    if idcg_at_k == 0.0:
        return 0.0

    return dcg_at_k / idcg_at_k