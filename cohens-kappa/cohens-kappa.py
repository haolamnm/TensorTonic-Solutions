import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    rater1 = np.asarray(rater1, dtype=int)
    rater2 = np.asarray(rater2, dtype=int)
    n = len(rater1)

    po = np.mean(rater1 == rater2)

    labels = np.union1d(rater1, rater2)
    pe = sum(
        np.sum(rater1 == k) * np.sum(rater2 == k)
        for k in labels
    ) / (n ** 2)

    score = (po - pe) / (1 - pe)
    return 1.0 if pe == 1 else score