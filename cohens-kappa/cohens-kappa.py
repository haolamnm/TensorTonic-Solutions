import numpy as np
from collections import Counter

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    label1 = set(rater1)
    label2 = set(rater2)
    n = len(rater1)
    print(n)

    agree = 0
    for r1, r2 in zip(rater1, rater2):
        agree += (r1 == r2)
    po = agree / n

    cnt1 = Counter(rater1)
    cnt2 = Counter(rater2)
    print(cnt1, cnt2)

    pel = []
    for label in label1 | label2:
       pel.append((cnt1[label] * cnt2[label]) / (n**2))
    print(pel)
    pe = np.sum(pel)

    print(po, pe)
    if pe == 1:
        return 1

    return (po - pe) / (1 - pe)