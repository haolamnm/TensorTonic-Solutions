import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    if not y_true_list or not y_score_list:
        return 0.0

    APs = []

    # each query has different length
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)

        n = len(y_true)
        _k = k if k and k <= n else n

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order][:_k]

        cum_hits = np.cumsum(y_true_sorted)
        ranks = np.arange(1, _k + 1)
        precision_at_k = cum_hits / ranks

        total_rel = np.sum(y_true)
        ap = np.sum(precision_at_k * y_true_sorted) / max(total_rel, 1)
        APs.append(ap)

    APs = np.array(APs)
    return float(np.mean(APs)), APs