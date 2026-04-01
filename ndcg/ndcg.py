import math
import numpy as np

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    if k < 1:
        raise ValueError

    if k > len(relevance_scores):
        k = len(relevance_scores)
    
    rel_base = np.asarray(relevance_scores, dtype=float)
    rel_sorted = np.sort(rel_base)[::-1]

    rel_base_at_k = rel_base[:k]
    rel_sorted_at_k = rel_sorted[:k]
    indices = np.arange(1, k + 1)

    deno = np.log2(indices + 1)
    dcg_at_k = np.sum((np.exp2(rel_base_at_k) - 1) / deno)
    idcg_at_k = np.sum((np.exp2(rel_sorted_at_k) - 1) / deno)

    if idcg_at_k == 0.0:
        return 0.0

    ncdg_at_k = dcg_at_k / idcg_at_k
    return float(ncdg_at_k)